import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import N_LABELS

CONT_ID = 16384


def import_arch(src_dir):
    for name in ["model_arch", "attention", "rope"]:
        sys.modules.pop(name, None)
    sys.path.insert(0, src_dir)
    mod = importlib.import_module("model_arch")
    sys.path.remove(src_dir)
    return mod


# phase 1 model without its lm head, ids [B, T] to hidden states [B, T, d]
class Backbone(nn.Module):

    def __init__(self, arch, model_dir, bidirectional=False):
        super().__init__()
        self.arch = arch
        mod = import_arch(model_dir)
        if arch == "arch1":
            cfg = mod.LMConfig(vocab_size=16384, d_model=768, n_layers=12, n_heads=12, n_kv_heads=12, d_ff=3072,
                               max_seq_len=1024, dropout=0.1, pos_emb_type="sinusoidal", norm_first=True)
        elif arch == "arch2":
            cfg = mod.LMConfig(vocab_size=16384, d_model=640, n_layers=12, n_heads=10, n_kv_heads=2, d_ff=1706,
                               max_seq_len=1024, dropout=0.0)
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
            raise RuntimeError(f"{arch}: missing {missing[:5]} unexpected {unexpected[:5]}")
        if arch == "arch3":
            model.dec_blocks = nn.ModuleList()
            del model.ln_f
        model.head = nn.Identity()
        self.model = model
        self.d_model = cfg.d_model
        self.prefix = []
        if arch == "arch3":
            self.prefix = [CONT_ID]
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
    return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), pos]


def mean_nonpad(hidden_states, attention_mask):
    m = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * m).sum(1) / m.sum(1).clamp(min=1)


# a4 q2 classifier, last token for causal models and mean for bidirectional ones
class PooledClassifier(nn.Module):

    def __init__(self, backbone, hidden_size, num_classes=2, pool="last"):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        self.pool = pool

    def features(self, input_ids):
        attention_mask = torch.ones_like(input_ids)
        h = self.backbone(input_ids).float()
        if self.pool == "last":
            return last_nonpad(h, attention_mask)
        return mean_nonpad(h, attention_mask)

    def forward(self, input_ids):
        return self.classifier(self.features(input_ids))


def make_classifier(arch, model_dir, bidirectional=False, num_classes=N_LABELS):
    bb = Backbone(arch, model_dir, bidirectional)
    pool = "last"
    if bb.bidirectional:
        pool = "mean"
    return PooledClassifier(bb, bb.d_model, num_classes, pool)


class BahdanauAttention(nn.Module):
    def __init__(self, dec_hidden, enc_hidden, attn_dim=128):
        super().__init__()
        self.W_a = nn.Linear(dec_hidden, attn_dim, bias=False)
        self.U_a = nn.Linear(enc_hidden, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        q = self.W_a(dec_hidden).unsqueeze(1)
        k = self.U_a(enc_outputs)
        energy = self.v_a(torch.tanh(q + k))
        scores = energy.squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        # matmul so one case broadcasts over the 100 label queries
        context = torch.matmul(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights


# all chunks of one case to 100 logits, "attn" has one query per label over every token
class CaseModel(nn.Module):

    def __init__(self, clf, head, prior):
        super().__init__()
        self.clf = clf
        self.head = head
        d = clf.backbone.d_model
        if head == "attn":
            self.queries = nn.Parameter(torch.randn(N_LABELS, d) * 0.02)
            self.attn = BahdanauAttention(d, d, 128)
            self.out_w = nn.Parameter(torch.randn(N_LABELS, d) * 0.02)
            self.out_b = nn.Parameter(torch.zeros(N_LABELS))
        # start every label at its training frequency
        b = torch.tensor(np.log(prior / (1 - prior)), dtype=torch.float32)
        if head == "attn":
            self.out_b.data.copy_(b)
        else:
            self.clf.classifier.bias.data.copy_(b)

    def forward(self, x):
        if self.head == "attn":
            h = self.clf.backbone(x).float().reshape(1, -1, self.out_w.shape[1])
            ctx, _ = self.attn(self.queries, h)
            return ((ctx * self.out_w).sum(-1) + self.out_b).unsqueeze(0)
        v = self.clf.features(x)
        return self.clf.classifier(v.mean(0, keepdim=True))


def asl_loss(z, y, gamma_neg=4.0, clip=0.05, eps=1e-8):
    # asymmetric loss, easy negatives are shifted by clip and down-weighted
    p = torch.sigmoid(z)
    p_neg = (p - clip).clamp(min=0)
    pos = y * torch.log(p.clamp(min=eps))
    neg = (1 - y) * torch.log((1 - p_neg).clamp(min=eps)) * p_neg ** gamma_neg
    return -(pos + neg).mean()


def case_loss(z, y, kind):
    if kind == "asl":
        return asl_loss(z, y)
    return F.binary_cross_entropy_with_logits(z, y)
