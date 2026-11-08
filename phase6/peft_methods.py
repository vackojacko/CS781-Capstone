# lorma (my phase 2 layer), lora (peft), pfeiffer and houlsby adapters on a frozen phase 1 backbone
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model

RANK = 16
ALPHA = 16
DROPOUT = 0.05
BOTTLENECK = 64
LINEARS = r".*blocks\.\d+\.(attn|ffn)\.(q_proj|k_proj|v_proj|kv_down_proj|k_up_proj|v_up_proj|out_proj|gate_proj|up_proj|down_proj|net\.0|net\.2)"


class LoRMALinear(nn.Module):
    # W' = (I + (alpha / r) B A) W0, A starts at zero so the layer starts as W0
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


class Adapter(nn.Module):
    # pfeiffer bottleneck after the ffn: y + up(gelu(down(y))), up starts at zero
    def __init__(self, ffn, d_model, bottleneck):
        super().__init__()
        self.ffn = ffn
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        y = self.ffn(x)
        return y + self.up(F.gelu(self.down(y)))


class HoulsbyAttention(nn.Module):
    # the same bottleneck after attention
    def __init__(self, attn, d_model, bottleneck):
        super().__init__()
        self.attn = attn
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        y = self.attn(x)
        return y + self.up(F.gelu(self.down(y)))


def get_blocks(backbone):
    if backbone.arch == "arch3":
        return backbone.model.enc_blocks
    return backbone.model.blocks


def add_lorma(blocks):
    for block in blocks:
        for module in list(block.modules()):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(module, name, LoRMALinear(child, RANK, ALPHA, DROPOUT))


def add_method(backbone, method):
    for p in backbone.parameters():
        p.requires_grad = False
    if method == "lorma":
        add_lorma(get_blocks(backbone))
        return backbone
    if method == "lora":
        config = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT, target_modules=LINEARS)
        return inject_adapter_in_model(config, backbone)
    for blk in get_blocks(backbone):
        blk.ffn = Adapter(blk.ffn, backbone.d_model, BOTTLENECK)
        if method == "houlsby":
            blk.attn = HoulsbyAttention(blk.attn, backbone.d_model, BOTTLENECK)
    return backbone
