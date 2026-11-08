# phase 1 models as sentence encoders (backbone loader and piece tokenizer from my phase 2 code)
import importlib
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn

BPE_PATH = "tokenizer/legal_bpe_16384.json"
KIT_DIR = "phase2_kit"  # arch1/ arch2/ arch3/ folders with model_arch.py, attention.py, rope.py, model.pt
ARCH4_MNTP_DIR = "arch4_mntp"  # bidirectional arch4 from the phase 2 MNTP adaptation
ARCH4_MNTP_P3_DIR = "arch4_mntp_p3"  # same, adapted longer incl. phase 3 training text

PIECE_RE = re.compile(r"[^\W\d_]+|\d+|_|[^\w\s]")

ARCH_CONFIG = {
    "arch1": dict(vocab_size=16384, d_model=768, n_layers=12, n_heads=12, n_kv_heads=12, d_ff=3072, max_seq_len=1024,
                  dropout=0.1, pos_emb_type="sinusoidal", norm_first=True),
    "arch2": dict(vocab_size=16384, d_model=640, n_layers=12, n_heads=10, n_kv_heads=2, d_ff=1706, max_seq_len=1024, dropout=0.0),
}
WINDOW = {"arch1": 1024, "arch2": 1024, "arch3": 704, "arch4": 1024}
CONT_ID = 16384


class PieceTokenizer:
    # my A1 BPE applied to the letter / digit / punctuation pieces of each word
    # only the last piece of a word gets the end-of-word marker
    def __init__(self, bpe):
        self.bpe = bpe
        bpe._rebuild_caches()
        self.cache = {}

    def encode_piece(self, s, eow):
        key = (s, eow)
        if key in self.cache:
            return self.cache[key]
        toks = []
        for b in s.encode("utf-8"):
            toks.append(b + 4)
        if eow:
            toks.append(self.bpe.eow_id)
        rank = self.bpe._merge_rank
        while len(toks) >= 2:
            best = None
            bi = -1
            for i in range(len(toks) - 1):
                r = rank.get((toks[i], toks[i + 1]))
                if r is not None and (best is None or r[0] < best[0]):
                    best = r
                    bi = i
            if bi < 0:
                break
            toks = toks[:bi] + [best[1]] + toks[bi + 2:]
        self.cache[key] = toks
        return toks

    def encode(self, text):
        ids = []
        for word in text.split():
            pieces = PIECE_RE.findall(word)
            for k in range(len(pieces)):
                ids += self.encode_piece(pieces[k], k == len(pieces) - 1)
        return np.array(ids, dtype=np.int64)


def load_tokenizer(bpe_path=BPE_PATH):
    sys.path.insert(0, os.path.dirname(os.path.abspath(bpe_path)))
    from bpe_tokenizer import BPETokenizer
    return PieceTokenizer(BPETokenizer.load(bpe_path))


def import_arch(src_dir):
    for name in ["model_arch", "attention", "rope"]:
        sys.modules.pop(name, None)
    sys.path.insert(0, src_dir)
    try:
        return importlib.import_module("model_arch")
    finally:
        sys.path.remove(src_dir)


class Backbone(nn.Module):
    # phase 1 model without its LM head: ids (B, T) to hidden states (B, T, d)
    def __init__(self, arch, model_dir, bidirectional=False, dropout=None):
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
            # only the encoder is used
            model.dec_blocks = nn.ModuleList()
            del model.ln_f
        model.head = nn.Identity()
        self.model = model
        self.d_model = cfg.d_model
        if arch == "arch3":
            self.prefix = [CONT_ID]
        else:
            self.prefix = []
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


def get_blocks(bb):
    if bb.arch == "arch3":
        return bb.model.enc_blocks
    return bb.model.blocks
