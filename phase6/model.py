# sentence salience model: phase 1 backbone, mean vector per sentence, bilstm (my A2), one logit per sentence
import importlib
import json
import os
import random
import sys

import torch
import torch.nn as nn

import peft_methods

MODEL_DIRS = {"arch1": "models/arch1", "arch2": "models/arch2", "arch3": "models/arch3", "arch4": "models/arch4_mntp"}

WINDOW = {"arch1": 1024, "arch2": 1024, "arch3": 704, "arch4": 1024}
CONT_ID = 16384
GRAD_WINDOWS = 3
HIDDEN = 256


def import_arch(src_dir):
    for name in ("model_arch", "attention", "rope"):
        sys.modules.pop(name, None)
    sys.path.insert(0, src_dir)
    mod = importlib.import_module("model_arch")
    sys.path.remove(src_dir)
    return mod


def load_config(model_dir):
    mc = json.load(open(os.path.join(model_dir, "metrics.json")))["model_config"]
    for k in mc:
        if isinstance(mc[k], list):
            mc[k] = tuple(mc[k])
    return mc


def make_config(mod, arch, model_dir):
    if arch == "arch1":
        return mod.LMConfig(vocab_size=16384, d_model=768, n_layers=12, n_heads=12, n_kv_heads=12, d_ff=3072,
                            max_seq_len=1024, dropout=0.1, pos_emb_type="sinusoidal", norm_first=True)
    if arch == "arch2":
        return mod.LMConfig(vocab_size=16384, d_model=640, n_layers=12, n_heads=10, n_kv_heads=2, d_ff=1706,
                            max_seq_len=1024, dropout=0.0)
    return mod.LMConfig(**load_config(model_dir))


class Backbone(nn.Module):
    # phase 1 model without its lm head, ids (B, T) to hidden states (B, T, d)
    def __init__(self, arch, model_dir, bidirectional=False, dropout=None):
        super().__init__()
        self.arch = arch
        mod = import_arch(model_dir)
        cfg = make_config(mod, arch, model_dir)
        if arch == "arch3":
            model = mod.EncoderDecoderLM(cfg)
        else:
            model = mod.DecoderLM(cfg)
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if k != "head.weight"]
        if arch == "arch4":
            comp = os.path.join(model_dir, "arch4_engram_compress.pt")
            if os.path.exists(comp):
                model.set_engram_compression(torch.load(comp, map_location="cpu", weights_only=False))
            missing = [k for k in missing if not k.endswith("engram.compress")]
        if missing or unexpected:
            raise RuntimeError(arch + ": missing " + str(missing[:5]) + " unexpected " + str(unexpected[:5]))
        if arch == "arch3":
            # keep only the encoder
            model.dec_blocks = nn.ModuleList()
            del model.ln_f
        model.head = nn.Identity()
        self.model = model
        self.d_model = cfg.d_model
        self.prefix = []
        if arch == "arch3":
            self.prefix = [CONT_ID]
        if dropout is not None:
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = dropout
        if bidirectional and arch != "arch3":
            for blk in model.blocks:
                blk.attn.causal_mask.fill_(True)
                if hasattr(blk.attn, "use_sdpa"):
                    blk.attn.use_sdpa = False

    def forward(self, ids):
        if self.arch == "arch3":
            return self.model.encode(ids)
        return self.model(ids)


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2g = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.h2g = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x_t, state):
        h_prev, c_prev = state[0], state[1]
        gates = self.x2g(x_t) + self.h2g(h_prev)
        f_pre, i_pre, g_pre, o_pre = gates.chunk(4, dim=-1)
        f_t = torch.sigmoid(f_pre)
        i_t = torch.sigmoid(i_pre)
        g_t = torch.tanh(g_pre)
        o_t = torch.sigmoid(o_pre)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return (h_t, c_t)


class RecurrentEncoder(nn.Module):
    def __init__(self, cell_cls, input_size, hidden_size, num_layers=1, bidirectional=False, is_lstm=False):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.is_lstm = is_lstm
        self.num_directions = 1
        if self.bidirectional:
            self.num_directions = 2
        self.cells = nn.ModuleList()
        for l in range(self.num_layers):
            in_dim = input_size
            if l > 0:
                in_dim = self.hidden_size * self.num_directions
            for _dir in range(self.num_directions):
                self.cells.append(cell_cls(in_dim, hidden_size))

    def _run_direction(self, cell, x, reverse):
        batch_size, seq_len, _ = x.shape
        device = x.device
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        c = None
        if self.is_lstm:
            c = torch.zeros(batch_size, self.hidden_size).to(device)
        steps = range(seq_len)
        if reverse:
            steps = reversed(range(seq_len))
        outputs = [None] * seq_len
        for t in steps:
            x_t = x[:, t, :]
            if self.is_lstm:
                h, c = cell(x_t, (h, c))
            else:
                h = cell(x_t, h)
            outputs[t] = h
        stacked_outputs = torch.stack(outputs, dim=1)
        if self.is_lstm:
            return (stacked_outputs, (h, c))
        return (stacked_outputs, h)

    def forward(self, x):
        layer_input = x
        cell_idx = 0
        final_h = []
        for l in range(self.num_layers):
            dir_outputs, dir_finals = [], []
            for d in range(self.num_directions):
                cell = self.cells[cell_idx]
                cell_idx = cell_idx + 1
                is_reverse = self.bidirectional and d == 1
                out, final = self._run_direction(cell, layer_input, is_reverse)
                dir_outputs.append(out)
                if self.is_lstm:
                    dir_finals.append(final[0])
                else:
                    dir_finals.append(final)
            layer_input = torch.cat(dir_outputs, dim=-1)
            final_h = torch.cat(dir_finals, dim=-1)
        return (layer_input, final_h)


def make_windows(toks, core, total):
    # sentences i..j get their vectors from this window, lo..i and j..hi are only context
    windows, i, n = [], 0, len(toks)
    while i < n:
        j, used = i, 0
        while j < n and (j == i or used + len(toks[j]) <= core):
            used += len(toks[j])
            j += 1
        lo, hi, budget = i, j, total - used
        grew = True
        while budget > 0 and grew:
            grew = False
            if lo > 0 and len(toks[lo - 1]) <= budget:
                lo -= 1
                budget -= len(toks[lo])
                grew = True
            if hi < n and len(toks[hi]) <= budget:
                budget -= len(toks[hi])
                hi += 1
                grew = True
        windows.append((lo, i, j, hi))
        i = j
    return windows


class SalienceModel(nn.Module):
    def __init__(self, backbone, hidden=HIDDEN):
        super().__init__()
        self.backbone = backbone
        self.window = WINDOW[backbone.arch] - len(backbone.prefix)
        self.lstm = RecurrentEncoder(LSTMCell, backbone.d_model + 2, hidden, num_layers=1, bidirectional=True, is_lstm=True)
        self.out = nn.Linear(2 * hidden, 1)
        self.drop = nn.Dropout(0.2)
        self.fp16 = backbone.arch != "arch1"

    def encode(self, toks):
        device = self.out.weight.device
        vecs = [None] * len(toks)
        windows = make_windows(toks, self.window, self.window)
        # only GRAD_WINDOWS random windows keep gradients, so memory per judgment stays bounded
        with_grad = set(range(len(windows)))
        if self.training and len(windows) > GRAD_WINDOWS:
            with_grad = set(random.sample(range(len(windows)), GRAD_WINDOWS))
        use_amp = device.type == "cuda" and self.fp16
        for w in range(len(windows)):
            lo, i, j, hi = windows[w]
            ids = list(self.backbone.prefix)
            for s in range(lo, hi):
                ids += toks[s]
            x = torch.tensor([ids]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                if w in with_grad:
                    h = self.backbone(x)[0]
                else:
                    with torch.no_grad():
                        h = self.backbone(x)[0]
            h = h.float()
            pos = len(self.backbone.prefix)
            for s in range(lo, hi):
                if i <= s < j:
                    vecs[s] = h[pos:pos + len(toks[s])].mean(dim=0)
                pos += len(toks[s])
        return torch.stack(vecs)

    def forward(self, toks):
        v = self.drop(self.encode(toks))
        n = len(toks)
        pos = torch.arange(n).float().to(v.device) / max(n - 1, 1)
        extra = torch.stack([pos, pos ** 2], dim=1)
        h, _ = self.lstm(torch.cat([v, extra], dim=1).unsqueeze(0))
        return self.out(self.drop(h[0])).squeeze(1)


def build(arch, method, bidirectional=False):
    backbone = Backbone(arch, MODEL_DIRS[arch], bidirectional=bidirectional, dropout=0.1)
    backbone = peft_methods.add_method(backbone, method)
    return SalienceModel(backbone)
