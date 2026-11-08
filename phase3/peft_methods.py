# parameter-efficient fine-tuning of the backbone
# lorma is my phase 2 layer, lora and ia3 come from the peft library
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, IA3Config, inject_adapter_in_model

from backbone import get_blocks

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LINEARS = r".*blocks\.\d+\.(attn|ffn)\.(q_proj|k_proj|v_proj|kv_down_proj|k_up_proj|v_up_proj|out_proj|gate_proj|up_proj|down_proj|net\.0|net\.2)"
IA3_LAYERS = r".*blocks\.\d+\.(attn\.(k_proj|v_proj|k_up_proj|v_up_proj)|ffn\.(down_proj|net\.2))"
IA3_FFN = r".*blocks\.\d+\.ffn\.(down_proj|net\.2)"


class LoRMALinear(nn.Module):
    # LoRMA+ (Bihany, Patel, Modi 2025): W' = (I + (alpha / r) B A) W0, computed as h + s B(A h) with h = W0 x
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


def add_peft(backbone, method):
    for p in backbone.parameters():
        p.requires_grad = False
    if method == "lorma":
        # every linear layer of every block
        for block in get_blocks(backbone):
            for module in list(block.modules()):
                for name, child in list(module.named_children()):
                    if isinstance(child, nn.Linear):
                        setattr(module, name, LoRMALinear(child, LORA_R, LORA_ALPHA, LORA_DROPOUT))
        return backbone
    if method == "lora":
        config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=LINEARS)
    elif method == "ia3":
        config = IA3Config(target_modules=IA3_LAYERS, feedforward_modules=IA3_FFN)
    else:
        raise ValueError("unknown method " + method)
    return inject_adapter_in_model(config, backbone)
