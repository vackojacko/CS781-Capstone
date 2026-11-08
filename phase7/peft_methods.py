import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dense_model import get_blocks

IA3_LAYERS = r".*blocks\.\d+\.(attn\.(k_proj|v_proj|k_up_proj|v_up_proj)|ffn\.(down_proj|net\.2))"
IA3_FFN = r".*blocks\.\d+\.ffn\.(down_proj|net\.2)"


# lorma+ from a4 q4
class LoRMALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        d_out = base.out_features
        B = torch.empty(d_out, rank)
        A = torch.empty(rank, d_out)
        bound = 1 / math.sqrt(d_out)
        A.uniform_(-bound, bound)
        B.uniform_(-bound, bound)
        B.zero_()
        self.B = nn.Parameter(B)
        self.A = nn.Parameter(A)
        self.scale = alpha / rank

    def forward(self, x):
        h = F.linear(x, self.base.weight)
        h = h + self.scale * F.linear(F.linear(h, self.A), self.B)
        if self.base.bias is None:
            return h
        return h + self.base.bias


def nf4_levels():
    # the 16 nf4 levels in [-1, 1]
    normal = torch.distributions.Normal(0.0, 1.0)
    offset = 0.9677083
    pos = normal.icdf(torch.linspace(offset, 0.5, 9)[:-1])
    neg = -normal.icdf(torch.linspace(offset, 0.5, 8)[:-1])
    v = torch.cat([pos, torch.zeros(1), neg])
    return (v / v.abs().max()).sort().values


def nf4_codes(w, block_size=64):
    flat = w.detach().float().reshape(-1)
    pad = (-flat.numel()) % block_size
    blocks = F.pad(flat, (0, pad)).reshape(-1, block_size)
    absmax = blocks.abs().amax(1, keepdim=True).clamp(min=1e-12)
    levels = nf4_levels().to(flat.device)
    idx = ((blocks / absmax).unsqueeze(-1) - levels).abs().argmin(-1)
    return idx.to(torch.uint8), absmax.squeeze(1)


# qlora from a4 q5
class QLoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, block_size=64):
        super().__init__()
        idx, absmax = nf4_codes(base.weight, block_size)
        idx = idx.reshape(-1)
        self.register_buffer("packed", (idx[0::2] << 4) | idx[1::2])
        self.register_buffer("absmax", absmax.half())
        self.register_buffer("levels", nf4_levels())
        self.shape = tuple(base.weight.shape)
        self.block = block_size
        if base.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank

    def weight(self, dtype):
        idx = torch.stack([self.packed >> 4, self.packed & 15], 1).reshape(-1).long()
        w = (self.levels[idx].reshape(-1, self.block) * self.absmax.float().unsqueeze(1)).reshape(-1)
        n = self.shape[0] * self.shape[1]
        return w[:n].reshape(self.shape).to(dtype)

    def forward(self, x):
        bias = None
        if self.bias is not None:
            bias = self.bias.to(x.dtype)
        h = F.linear(x, self.weight(x.dtype), bias)
        return h + self.scaling * F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))


def replace_linears(blocks, method, rank, alpha):
    for blk in blocks:
        for mod in list(blk.modules()):
            for name, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    if method == "lorma":
                        setattr(mod, name, LoRMALinear(child, rank, alpha))
                    else:
                        setattr(mod, name, QLoRALinear(child, rank, alpha))


def unfreeze_top(bb, k):
    # only the top k blocks and the final norm train
    blocks = get_blocks(bb)
    if k >= len(blocks):
        for p in bb.parameters():
            p.requires_grad = True
        return
    for blk in list(blocks)[len(blocks) - k:]:
        for p in blk.parameters():
            p.requires_grad = True
    if k > 0:
        for name, p in bb.named_parameters():
            if name.startswith("model.ln_f") or name.startswith("model.enc_norm"):
                p.requires_grad = True


def attach_method(clf, method, cfg):
    bb = clf.backbone
    for p in bb.parameters():
        p.requires_grad = False
    if method == "ia3":
        from peft import IA3Config, inject_adapter_in_model
        config = IA3Config(target_modules=IA3_LAYERS, feedforward_modules=IA3_FFN)
        inject_adapter_in_model(config, bb)
    elif method == "lorma":
        replace_linears(get_blocks(bb), "lorma", cfg["rank"], cfg["alpha"])
    elif method == "qlora":
        replace_linears(get_blocks(bb), "qlora", cfg["rank"], cfg["alpha"])
    elif method == "partial":
        unfreeze_top(bb, cfg["top_blocks"])
    else:
        raise ValueError(method)
    for p in clf.classifier.parameters():
        p.requires_grad = True
    return clf
