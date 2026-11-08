# phase 1 models without their lm head (same loader as phase 2): ids [B, T] to hidden states [B, T, d]
import importlib
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

ARCH_CONFIG = {
    "arch1": dict(vocab_size=16384, d_model=768, n_layers=12, n_heads=12, n_kv_heads=12, d_ff=3072, max_seq_len=1024,
                  dropout=0.1, pos_emb_type="sinusoidal", norm_first=True),
    "arch2": dict(vocab_size=16384, d_model=640, n_layers=12, n_heads=10, n_kv_heads=2, d_ff=1706, max_seq_len=1024,
                  dropout=0.0),
}
CONT_ID = 16384


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_arch(src_dir):
    # model_arch.py, attention.py and rope.py of each phase 1 model sit next to its checkpoint
    for name in ("model_arch", "attention", "rope"):
        sys.modules.pop(name, None)
    sys.path.insert(0, src_dir)
    mod = importlib.import_module("model_arch")
    sys.path.remove(src_dir)
    return mod


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
        missing = [k for k in missing if k != "head.weight"]
        if arch == "arch4":
            comp = os.path.join(model_dir, "arch4_engram_compress.pt")
            if os.path.exists(comp):
                model.set_engram_compression(torch.load(comp, map_location="cpu", weights_only=False))
            missing = [k for k in missing if not k.endswith("engram.compress")]
        if missing or unexpected:
            raise RuntimeError(arch + ": missing " + str(missing[:5]) + " unexpected " + str(unexpected[:5]))
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
        # bidirectional arch 4 (mntp): no causal mask
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
