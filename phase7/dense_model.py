import importlib
import json
import os
import sys

import torch
import torch.nn as nn

ARCH_CONFIG = {
    "arch1": dict(vocab_size=16384, d_model=768, n_layers=12, n_heads=12, n_kv_heads=12, d_ff=3072, max_seq_len=1024,
                  dropout=0.1, pos_emb_type="sinusoidal", norm_first=True),
    "arch2": dict(vocab_size=16384, d_model=640, n_layers=12, n_heads=10, n_kv_heads=2, d_ff=1706, max_seq_len=1024, dropout=0.0),
}
WINDOW = {"arch1": 1024, "arch2": 1024, "arch3": 704, "arch4": 1024}
CONT_ID = 16384


def import_arch(model_dir):
    # load the phase 1 model_arch.py saved in model_dir
    for name in ("model_arch", "attention", "rope"):
        sys.modules.pop(name, None)
    sys.path.insert(0, model_dir)
    mod = importlib.import_module("model_arch")
    sys.path.remove(model_dir)
    return mod


# phase 1 model without its lm head
class Backbone(nn.Module):
    def __init__(self, arch, model_dir, bidirectional=False):
        super().__init__()
        self.arch = arch
        mod = import_arch(model_dir)
        if arch in ARCH_CONFIG:
            cfg = mod.LMConfig(**ARCH_CONFIG[arch])
        else:
            mc = json.load(open(os.path.join(model_dir, "metrics.json")))["model_config"]
            for k in mc:
                if isinstance(mc[k], list):
                    mc[k] = tuple(mc[k])
            cfg = mod.LMConfig(**mc)
        if arch == "arch3":
            model = mod.EncoderDecoderLM(cfg)
        else:
            model = mod.DecoderLM(cfg)
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if arch == "arch4":
            comp = os.path.join(model_dir, "arch4_engram_compress.pt")
            if os.path.exists(comp):
                model.set_engram_compression(torch.load(comp, map_location="cpu", weights_only=False))
        left = []
        for k in missing:
            if k == "head.weight":
                continue
            if arch == "arch4" and k.endswith("engram.compress"):
                continue
            left.append(k)
        missing = left
        if len(missing) > 0 or len(unexpected) > 0:
            raise RuntimeError(arch + " checkpoint does not match: " + str(missing[:5]) + " " + str(unexpected[:5]))
        if arch == "arch3":
            model.dec_blocks = nn.ModuleList()
            del model.ln_f
        model.head = nn.Identity()
        self.model = model
        self.d_model = cfg.d_model
        if arch == "arch3":
            self.prefix = [CONT_ID]
        else:
            self.prefix = []
        self.bidirectional = bidirectional or arch == "arch3"
        if bidirectional and arch != "arch3":
            for blk in model.blocks:
                blk.attn.causal_mask.fill_(True)
                if hasattr(blk.attn, "use_sdpa"):
                    blk.attn.use_sdpa = False

    def forward(self, ids):
        if self.arch == "arch3":
            return self.model.encode(ids)
        return self.model(ids)


def get_blocks(bb):
    if bb.arch == "arch3":
        return bb.model.enc_blocks
    return bb.model.blocks


def last_nonpad(hidden_states, attention_mask):
    T = attention_mask.shape[1]
    pos = T - 1 - attention_mask.long().flip(dims=[1]).argmax(dim=1)
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, pos]


def mean_nonpad(hidden_states, attention_mask):
    m = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * m).sum(1) / m.sum(1).clamp(min=1)


# classifier from a4 q2
class Classifier(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes=2, pool="last"):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        self.pool = pool

    def features(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        h = self.backbone(input_ids).float()
        if self.pool == "last":
            return last_nonpad(h, attention_mask)
        return mean_nonpad(h, attention_mask)

    def forward(self, input_ids=None, attention_mask=None):
        return self.classifier(self.features(input_ids, attention_mask))


def make_classifier(arch, model_dir, bidirectional=False, num_classes=2):
    bb = Backbone(arch, model_dir, bidirectional)
    pool = "last"
    if bb.bidirectional:
        pool = "mean"
    return Classifier(bb, bb.d_model, num_classes, pool)


def window_starts(n, length, stride):
    if n <= length:
        return [0]
    s = list(range(0, n - length + 1, stride))
    if s[-1] != n - length:
        s.append(n - length)
    return s


def case_chunks(ids, bb, k):
    # quarter overlap, keep the first k-1 chunks and the last one
    L = WINDOW[bb.arch] - len(bb.prefix)
    s = window_starts(len(ids), L, L * 3 // 4)
    if len(s) > k:
        s = s[:k - 1] + [s[-1]]
    chunks = []
    for a in s:
        chunks.append(list(bb.prefix) + ids[a:a + L].tolist())
    return chunks
