import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import get_blocks


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

    def __init__(self, block, adapter):
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, x, *args):
        return self.adapter(self.block(x, *args))


# B [d_out, r], A [r, d_out], init=True starts at the base weight
def make_lorma_factors(d_in, d_out, rank, init=False, generator=None, dtype=torch.float64, swap_ab=False, alpha=None):
    if alpha is None:
        alpha = rank
    scale = alpha / rank
    B = torch.empty(d_out, rank, dtype=dtype)
    A = torch.empty(rank, d_out, dtype=dtype)
    bound = 1 / math.sqrt(d_out)
    A.uniform_(-bound, bound, generator=generator)
    B.uniform_(-bound, bound, generator=generator)
    if init:
        if swap_ab:
            A.zero_()
        else:
            B.zero_()
    return B, A, scale


def lorma_dense(x, frozen_weight, factors):
    # W' = (s B A + I) W0
    B, A, s = factors
    W = (s * B @ A + torch.eye(B.shape[0], dtype=B.dtype, device=B.device)) @ frozen_weight
    return x @ W.T


def lorma_reordered(x, frozen_weight, factors):
    # same function without the d x d product
    B, A, s = factors
    h = x @ frozen_weight.T
    return h + s * (h @ A.T) @ B.T


class LoRMALinear(nn.Module):

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


def check_lorma(input_dim=12, output_dim=16, rank=3):
    # dense and reordered must match in float64, and init must give the base
    g = torch.Generator().manual_seed(0)
    W = torch.randn(output_dim, input_dim, dtype=torch.float64, generator=g)
    x = torch.randn(5, input_dim, dtype=torch.float64, generator=g)
    B, A, s = make_lorma_factors(input_dim, output_dim, rank, init=False, generator=g, alpha=2 * rank)
    B.requires_grad_(True)
    A.requires_grad_(True)
    yd = lorma_dense(x, W, (B, A, s))
    gd = torch.autograd.grad(yd.sum(), (B, A))
    yr = lorma_reordered(x, W, (B, A, s))
    gr = torch.autograd.grad(yr.sum(), (B, A))
    assert torch.allclose(yd, yr)
    assert torch.allclose(gd[0], gr[0]) and torch.allclose(gd[1], gr[1])
    for swap in [False, True]:
        f0 = make_lorma_factors(input_dim, output_dim, rank, init=True, generator=g, swap_ab=swap)
        assert torch.allclose(lorma_reordered(x, W, f0), x @ W.T)
    print("LoRMA checks passed")


# the 16 nf4 levels in [-1, 1] with an exact 0
def nf4_levels():
    normal = torch.distributions.Normal(0.0, 1.0)
    offset = 0.9677083
    pos = normal.icdf(torch.linspace(offset, 0.5, 9)[:-1])
    neg = -normal.icdf(torch.linspace(offset, 0.5, 8)[:-1])
    v = torch.cat([pos, torch.zeros(1), neg])
    return (v / v.abs().max()).sort().values


def nf4_codes(w, block_size=64):
    # one level index per weight and one absmax per block
    flat = w.detach().float().reshape(-1)
    pad = (-flat.numel()) % block_size
    blocks = F.pad(flat, (0, pad)).reshape(-1, block_size)
    absmax = blocks.abs().amax(1, keepdim=True).clamp(min=1e-12)
    levels = nf4_levels().to(flat.device)
    idx = ((blocks / absmax).unsqueeze(-1) - levels).abs().argmin(-1)
    return idx.to(torch.uint8), absmax.squeeze(1)


def nf4_quantize(w, block_size=64):
    idx, absmax = nf4_codes(w, block_size)
    deq = nf4_levels().to(absmax.device)[idx.long()] * absmax.unsqueeze(1)
    return deq.reshape(-1)[: w.numel()].reshape(w.shape).to(w.dtype)


class QLoRALinear(nn.Module):

    def __init__(self, base, rank=8, alpha=16, block_size=64):
        super().__init__()
        idx, absmax = nf4_codes(base.weight, block_size)
        idx = idx.reshape(-1)
        # two 4-bit codes per byte
        self.register_buffer("packed", (idx[0::2] << 4) | idx[1::2])
        self.register_buffer("absmax", absmax.half())
        self.register_buffer("levels", nf4_levels())
        self.shape = tuple(base.weight.shape)
        self.block = block_size
        self.bias = None
        if base.bias is not None:
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


def check_nf4():
    lv = nf4_levels()
    assert len(lv) == 16 and (lv == 0).any() and lv.min() == -1 and lv.max() == 1
    torch.manual_seed(0)
    lin = nn.Linear(96, 80)
    q = QLoRALinear(lin, rank=4)
    x = torch.randn(3, 96)
    w_fake = nf4_quantize(lin.weight)
    # packed weights match fake quantisation, and B = 0 gives the quantised base
    assert torch.allclose(q.weight(torch.float32), w_fake, atol=1e-3)
    assert torch.allclose(q(x), F.linear(x, w_fake, lin.bias), atol=1e-3)
    err = ((w_fake - lin.weight).norm() / lin.weight.norm()).item()
    print(f"NF4 checks passed | relative error {err:.4f}")


def replace_linears(blocks, method, rank, alpha):
    for blk in blocks:
        for mod in list(blk.modules()):
            for name, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    if method == "lorma":
                        setattr(mod, name, LoRMALinear(child, rank, alpha))
                    else:
                        setattr(mod, name, QLoRALinear(child, rank, alpha))


def unfreeze_top(classifier, k):
    # the top k blocks and the final norm become trainable
    bb = classifier.backbone
    blocks = get_blocks(bb)
    for blk in list(blocks)[len(blocks) - k:]:
        for p in blk.parameters():
            p.requires_grad = True
    for name, p in bb.named_parameters():
        if name.startswith("model.ln_f") or name.startswith("model.enc_norm"):
            p.requires_grad = True


# method is "full", "adapter", "lorma", "qlora" or "partial", the head always stays trainable
def attach_peft(classifier, method, config=None):
    if config is None:
        config = {}
    bb = classifier.backbone
    blocks = get_blocks(bb)
    rank = config.get("rank", 8)
    alpha = config.get("alpha", 16)
    if method != "full":
        for p in bb.parameters():
            p.requires_grad = False
    if method == "adapter":
        for i in range(len(blocks)):
            blocks[i] = AdapterBlock(blocks[i], BottleneckAdapter(bb.d_model, config.get("bottleneck", 64)))
    elif method == "lorma":
        replace_linears(blocks, "lorma", rank, alpha)
    elif method == "qlora":
        replace_linears(blocks, "qlora", rank, alpha)
    elif method == "partial":
        unfreeze_top(classifier, config.get("top_blocks", 8))
    elif method != "full":
        raise ValueError(method)
    for p in classifier.classifier.parameters():
        p.requires_grad = True
    return classifier
