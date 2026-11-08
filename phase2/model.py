# phase 1 backbones with a BIOES token classification head, and sliding-window inference
import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import N_TAGS, window_starts

# folders with each phase 1 model: model_arch.py (its file from ../phase1), model.pt,
# and for arch3 and arch4 also metrics.json (arch4 also arch4_engram_compress.pt)
MODEL_DIRS = {"arch1": "models/arch1", "arch2": "models/arch2", "arch3": "models/arch3", "arch4": "models/arch4"}

ARCH_CONFIG = {
    "arch1": dict(vocab_size=16384, d_model=768, n_layers=12, n_heads=12, n_kv_heads=12, d_ff=3072, max_seq_len=1024,
                  dropout=0.1, pos_emb_type="sinusoidal", norm_first=True),
    "arch2": dict(vocab_size=16384, d_model=640, n_layers=12, n_heads=10, n_kv_heads=2, d_ff=1706, max_seq_len=1024, dropout=0.0),
}
WINDOW = {"arch1": 1024, "arch2": 1024, "arch3": 704, "arch4": 1024}
CONT_ID = 16384
SEP_ID = 3
# first HEADER tokens of the document are put before every window (parties, counsel, judges)
HEADER = 0


def window_prefix(bb, doc):
    if HEADER <= 0:
        return list(bb.prefix)
    return list(bb.prefix) + doc["ids"][:HEADER].tolist() + [SEP_ID]


def content_length(bb):
    n = WINDOW[bb.arch] - len(bb.prefix)
    if HEADER > 0:
        n -= HEADER + 1
    return n


def import_arch(src_dir):
    for name in ("model_arch", "attention", "rope"):
        sys.modules.pop(name, None)
    sys.path.insert(0, src_dir)
    try:
        return importlib.import_module("model_arch")
    finally:
        sys.path.remove(src_dir)


class Backbone(nn.Module):
    # phase 1 model without its LM head, maps ids (B, T) to hidden states (B, T, d)

    def __init__(self, arch, model_dir, bidirectional=False, dropout=None):
        super().__init__()
        self.arch = arch
        mod = import_arch(model_dir)
        if arch in ARCH_CONFIG:
            cfg = mod.LMConfig(**ARCH_CONFIG[arch])
        else:
            with open(os.path.join(model_dir, "metrics.json")) as f:
                mc = json.load(f)["model_config"]
            for k in mc:
                if isinstance(mc[k], list):
                    mc[k] = tuple(mc[k])
            cfg = mod.LMConfig(**mc)
        if arch == "arch3":
            model = mod.EncoderDecoderLM(cfg)
        else:
            model = mod.DecoderLM(cfg)
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
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
            # keep only the encoder of arch 3
            model.dec_blocks = nn.ModuleList()
            del model.ln_f
        model.head = nn.Identity()
        self.model = model
        self.d_model = cfg.d_model
        self.prefix = []
        if arch == "arch3":
            self.prefix = [CONT_ID]
        self.bidirectional = bidirectional or arch == "arch3"
        if dropout is not None:
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = dropout
        if bidirectional and arch != "arch3":
            # unmask attention of the decoder
            for blk in model.blocks:
                blk.attn.causal_mask.fill_(True)
                if hasattr(blk.attn, "use_sdpa"):
                    blk.attn.use_sdpa = False

    def forward(self, ids):
        if self.arch == "arch3":
            return self.model.encode(ids)
        return self.model(ids)


class TokenClassifier(nn.Module):

    def __init__(self, backbone, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(backbone.d_model, N_TAGS)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids):
        return self.head(self.drop(self.backbone(ids)))


def load_classifier(arch, model_dir, weights_path):
    model = TokenClassifier(Backbone(arch, model_dir))
    sd = torch.load(weights_path, map_location="cpu")
    for k in sd:
        sd[k] = sd[k].float()
    model.load_state_dict(sd)
    return model


@torch.no_grad()
def doc_logprobs(model, doc, device, batch=8):
    # each token keeps the window with the most context (both sides if bidirectional, left side if causal)
    bb = model.backbone
    pre = window_prefix(bb, doc)
    length = content_length(bb)
    stride = length // 2
    n = len(doc["ids"])
    out = np.zeros((n, N_TAGS), dtype=np.float32)
    if n == 0:
        return out
    best = np.full(n, -1, dtype=np.int64)
    starts = window_starts(n, length, stride)
    use_amp = device.startswith("cuda")
    model.eval()
    for k in range(0, len(starts), batch):
        chunk = starts[k:k + batch]
        L = min(length, n)
        rows = []
        for s in chunk:
            rows.append(pre + doc["ids"][s:s + L].tolist())
        ids = torch.tensor(rows, device=device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=use_amp):
            logits = model(ids)[:, len(pre):].float()
        lp = F.log_softmax(logits, -1).cpu().numpy()
        for s, row in zip(chunk, lp):
            pos = np.arange(L)
            if bb.bidirectional:
                ctx = np.minimum(pos, L - 1 - pos)
            else:
                ctx = pos
            g = s + pos
            take = ctx > best[g]
            out[g[take]] = row[take]
            best[g[take]] = ctx[take]
    return out
