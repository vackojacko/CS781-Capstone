# A4 q4 skeleton (peft from scratch), completed for my phase 1 models (nn.Linear instead of gpt-2 Conv1D)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import get_blocks


# 4.2 adapters
class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size, bottleneck=16):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states):
        return hidden_states + self.up(F.gelu(self.down(hidden_states)))


class AdapterBlock(nn.Module):
    # adapter on the output of a block
    def __init__(self, block, adapter):
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, x, *args):
        return self.adapter(self.block(x, *args))


# 4.4 lorma+: W' = (alpha/r * B A + I) W0
def make_lorma_factors(d_in, d_out, rank, init=False, generator=None, dtype=torch.float64, alpha=None):
    if alpha is None:
        alpha = rank
    scale = alpha / rank
    B = torch.empty(d_out, rank, dtype=dtype)
    A = torch.empty(rank, d_out, dtype=dtype)
    bound = 1 / math.sqrt(d_out)
    A.uniform_(-bound, bound, generator=generator)
    B.uniform_(-bound, bound, generator=generator)
    # init=True keeps the base output
    if init:
        B.zero_()
    return B, A, scale


class LoRMALinear(nn.Module):
    # frozen linear layer with the lorma+ update in the reordered form
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        B, A, s = make_lorma_factors(base.in_features, base.out_features, rank, init=True, dtype=torch.float32,
                                     alpha=alpha)
        self.B = nn.Parameter(B)
        self.A = nn.Parameter(A)
        self.scale = s

    def forward(self, x):
        h = F.linear(x, self.base.weight)
        h = h + self.scale * F.linear(F.linear(h, self.A), self.B)
        if self.base.bias is None:
            return h
        return h + self.base.bias


def unfreeze_top(classifier, k):
    # top k blocks and the final norm become trainable
    bb = classifier.backbone
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


def attach_peft(classifier, method, config=None):
    # method is full, adapter, lorma or unfreeze
    if config is None:
        config = {}
    bb = classifier.backbone
    blocks = get_blocks(bb)
    if method != "full":
        for p in bb.parameters():
            p.requires_grad = False
    if method == "adapter":
        for i in range(len(blocks)):
            blocks[i] = AdapterBlock(blocks[i], BottleneckAdapter(bb.d_model, config.get("bottleneck", 64)))
    elif method == "lorma":
        for blk in blocks:
            for mod in list(blk.modules()):
                for name, child in list(mod.named_children()):
                    if isinstance(child, nn.Linear):
                        setattr(mod, name, LoRMALinear(child, config.get("rank", 8), config.get("alpha", 16)))
    elif method == "unfreeze":
        unfreeze_top(classifier, config.get("start_blocks", 0))
    elif method != "full":
        raise ValueError(method)
    for p in classifier.classifier.parameters():
        p.requires_grad = True
    return classifier
