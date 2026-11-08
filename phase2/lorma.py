# LoRA and LoRMA layers written by hand, and freezing the backbone around them
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    # h = W0 x + (alpha / r) B A x, B starts at zero

    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)


class LoRMALinear(nn.Module):
    # LoRMA+: W' = (I + (alpha / r) B A) W0, computed as h + s B(A h) with h = W0 x
    # A starts at zero so the layer starts as the frozen W0

    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        d = base.out_features
        self.A = nn.Parameter(torch.zeros(r, d))
        self.B = nn.Parameter(torch.empty(d, r))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.linear(x, self.base.weight)
        h = h + self.scale * F.linear(F.linear(self.drop(h), self.A), self.B)
        if self.base.bias is None:
            return h
        return h + self.base.bias


def get_blocks(bb):
    if bb.arch == "arch3":
        return bb.model.enc_blocks
    return bb.model.blocks


def apply_method(model, method, r=16, alpha=16, dropout=0.05):
    # freezes the backbone and wraps every linear layer of every block; the token head stays trainable
    bb = model.backbone
    if method == "full":
        return model
    for p in bb.parameters():
        p.requires_grad = False
    if method == "lora":
        layer_class = LoRALinear
    elif method == "lorma":
        layer_class = LoRMALinear
    else:
        raise ValueError(method)
    for blk in get_blocks(bb):
        for name, mod in list(blk.named_modules()):
            for child_name, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(mod, child_name, layer_class(child, r, alpha, dropout))
    return model
