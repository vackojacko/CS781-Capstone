# small Arch 4 for the ablations: latent attention, QK-Norm, gated heads and Engram memory, each with a switch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LMConfig:
    def __init__(self, vocab_size, d_model=640, n_layers=16, n_heads=10, kv_latent_dim=320, rope_dim=32,
                 d_ff=1712, max_seq_len=1024, dropout=0.0, engram_layers=(2, 8), engram_orders=(2, 3),
                 engram_heads=4, engram_head_dim=32, engram_rows=81920, engram_vocab=16384, qk_norm=True,
                 gated=True, latent=True):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.kv_latent_dim = kv_latent_dim
        self.rope_dim = rope_dim
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.engram_layers = engram_layers
        self.engram_orders = engram_orders
        self.engram_heads = engram_heads
        self.engram_head_dim = engram_head_dim
        self.engram_rows = engram_rows
        self.engram_vocab = engram_vocab
        self.qk_norm = qk_norm
        self.gated = gated
        self.latent = latent


def build_rope_cache(max_seq_len, d_head, base=10000.0):
    assert d_head % 2 == 0, "RoPE requires an even head dimension"
    inverse_frequencies = torch.zeros(d_head // 2)
    for i in range(d_head // 2):
        inverse_frequencies[i] = 1 / (base ** (2 * i / d_head))

    phi_values = torch.zeros(max_seq_len, d_head // 2)
    for i in range(max_seq_len):
        for j in range(d_head // 2):
            phi_values[i, j] = i * inverse_frequencies[j]

    angles = torch.cat([phi_values, phi_values], dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    T, d_head = cos.shape
    cos = cos.view(1, 1, T, d_head)
    sin = sin.view(1, 1, T, d_head)
    return x * cos + rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # computed in fp32 so x*x cannot overflow under fp16
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, kv_latent_dim, max_seq_len, rope_dim=32, dropout=0.0, qk_norm=True,
                 gated=True, latent=True):
        super().__init__()
        assert d_model % n_heads == 0
        assert 0 < kv_latent_dim < d_model

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rope_dim = rope_dim
        self.kv_latent_dim = kv_latent_dim
        self.attention_scale = 1.0 / math.sqrt(self.d_head + rope_dim)

        self.latent = latent
        if latent:
            self.q_proj = nn.Linear(d_model, n_heads * (self.d_head + rope_dim), bias=False)
            self.kv_down_proj = nn.Linear(d_model, kv_latent_dim + rope_dim, bias=False)
            self.kv_norm = RMSNorm(kv_latent_dim)
            self.k_up_proj = nn.Linear(kv_latent_dim, d_model, bias=False)
            self.v_up_proj = nn.Linear(kv_latent_dim, d_model, bias=False)
            qk_dim = self.d_head + rope_dim
        else:
            # standard MHA with RoPE on the full head
            self.rope_dim = self.d_head
            self.attention_scale = 1.0 / math.sqrt(self.d_head)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            qk_dim = self.d_head
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        if qk_norm:
            self.q_norm = RMSNorm(qk_dim)
            self.k_norm = RMSNorm(qk_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        if gated:
            self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        else:
            self.gate_proj = None
        self.use_sdpa = False

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)
        cos, sin = build_rope_cache(max_seq_len, self.rope_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, cache=None, pos0=0):
        B, T, C = x.shape
        H = self.n_heads
        D = self.d_head
        R = self.rope_dim

        if self.latent:
            q = self.q_proj(x).view(B, T, H, D + R).transpose(1, 2)
            z = self.kv_down_proj(x)
            c = z[..., :self.kv_latent_dim]
            k_rope = z[..., self.kv_latent_dim:]
            c = self.kv_norm(c)
            k = self.k_up_proj(c).view(B, T, H, D).transpose(1, 2)
            v = self.v_up_proj(c).view(B, T, H, D).transpose(1, 2)
            # one rope key shared by all heads
            k = torch.cat([k, k_rope.unsqueeze(1).expand(B, H, T, R)], dim=-1)

            q = self.q_norm(q)
            k = self.k_norm(k)
            cos = self.rope_cos[pos0:pos0 + T]
            sin = self.rope_sin[pos0:pos0 + T]
            q = torch.cat([q[..., :D], apply_rope(q[..., D:], cos, sin)], dim=-1)
            k = torch.cat([k[..., :D], apply_rope(k[..., D:], cos, sin)], dim=-1)
        else:
            q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
            k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
            v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
            q = self.q_norm(q)
            k = self.k_norm(k)
            cos = self.rope_cos[pos0:pos0 + T]
            sin = self.rope_sin[pos0:pos0 + T]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"] = k
            cache["v"] = v
        Tk = k.shape[2]
        mask = self.causal_mask[pos0:pos0 + T, :Tk]

        if self.use_sdpa:
            if pos0 == 0 and T == Tk:
                final = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attention_scale)
            else:
                final = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.attention_scale)
        else:
            scores = (q @ k.transpose(-2, -1)) * self.attention_scale
            scores = scores.masked_fill(mask == 0, float('-inf'))
            probs = torch.softmax(scores, dim=-1)
            probs = self.attn_dropout(probs)
            final = probs @ v

        if self.gate_proj is not None:
            final = final * torch.sigmoid(self.gate_proj(x)).view(B, T, H, D).transpose(1, 2)
        final = final.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(final)
        out = self.resid_dropout(out)
        return out


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


def primes_from(start, count):
    out = []
    n = start
    while len(out) < count:
        n += 1
        is_prime = True
        for p in range(2, int(n ** 0.5) + 1):
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            out.append(n)
    return out


class Engram(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.orders = tuple(cfg.engram_orders)
        self.heads = cfg.engram_heads
        n_tables = len(self.orders) * self.heads
        primes = primes_from(cfg.engram_rows + 101 * layer_idx, n_tables)
        offsets = [0]
        for p in primes[:-1]:
            offsets.append(offsets[-1] + p)
        g = torch.Generator().manual_seed(1000 + layer_idx)
        # odd multipliers for the hashes
        mult = torch.randint(1, 2 ** 30, (n_tables, max(self.orders)), generator=g) * 2 + 1
        self.register_buffer("primes", torch.tensor(primes, dtype=torch.long), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.register_buffer("mult", mult.long(), persistent=False)
        self.register_buffer("compress", torch.arange(cfg.vocab_size, dtype=torch.long), persistent=False)

        self.table = nn.Embedding(sum(primes), cfg.engram_head_dim)
        d_mem = n_tables * cfg.engram_head_dim
        self.w_k = nn.Linear(d_mem, cfg.d_model, bias=False)
        self.w_v = nn.Linear(d_mem, cfg.d_model, bias=False)
        self.norm_h = RMSNorm(cfg.d_model)
        self.norm_k = RMSNorm(cfg.d_model)
        self.norm_v = RMSNorm(cfg.d_model)
        self.dilation = max(self.orders)
        self.conv = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=4, groups=cfg.d_model,
                              dilation=self.dilation, bias=False)
        self.d_model = cfg.d_model

    def hash_indices(self, ids):
        c = self.compress[ids]
        B, T = c.shape
        # shifted[j] holds token t-j at position t
        shifted = [c]
        for j in range(1, max(self.orders)):
            shifted.append(F.pad(c, (j, 0), value=0)[:, :T])
        idx = []
        tbl = 0
        for n in self.orders:
            for _ in range(self.heads):
                h = shifted[0] * self.mult[tbl, 0]
                for j in range(1, n):
                    h = torch.bitwise_xor(h, shifted[j] * self.mult[tbl, j])
                idx.append(torch.remainder(h, self.primes[tbl]) + self.offsets[tbl])
                tbl += 1
        return torch.stack(idx, dim=-1)

    def forward(self, h, ids, cache=None):
        B, T, _ = h.shape
        keep = max(self.orders) - 1
        if cache is not None and cache.get("ids") is not None:
            full_ids = torch.cat([cache["ids"], ids], dim=1)
        else:
            full_ids = ids
        idx = self.hash_indices(full_ids)[:, -T:]
        if cache is not None:
            if keep > 0:
                cache["ids"] = full_ids[:, full_ids.shape[1] - keep:]
            else:
                cache["ids"] = full_ids[:, :0]
        e = self.table(idx).view(B, T, -1)
        k = self.w_k(e)
        v = self.w_v(e)
        s = (self.norm_h(h) * self.norm_k(k)).sum(-1, keepdim=True).float() / math.sqrt(self.d_model)
        alpha = torch.sigmoid(torch.sign(s) * torch.sqrt(s.abs() + 1e-6)).type_as(v)
        u = alpha * v
        y = self.norm_v(u).transpose(1, 2)
        pad = (self.conv.kernel_size[0] - 1) * self.dilation
        if cache is not None:
            hist = cache.get("conv")
            if hist is None:
                hist = y.new_zeros(B, y.shape[1], pad)
            y = torch.cat([hist.to(y.dtype), y], dim=2)
            cache["conv"] = y[:, :, -pad:]
        else:
            # left pad only, so the convolution is causal
            y = F.pad(y, (pad, 0))
        y = F.silu(self.conv(y)).transpose(1, 2)
        return y + u


class Block(nn.Module):
    def __init__(self, cfg, layer_idx, use_engram):
        super().__init__()
        if use_engram:
            self.engram = Engram(cfg, layer_idx)
        else:
            self.engram = None
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = MultiHeadLatentAttention(cfg.d_model, cfg.n_heads, cfg.kv_latent_dim, cfg.max_seq_len,
                                             rope_dim=cfg.rope_dim, dropout=cfg.dropout, qk_norm=cfg.qk_norm,
                                             gated=cfg.gated, latent=cfg.latent)
        self.ln2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x, ids, cache=None, pos0=0):
        engram_cache = None
        attn_cache = None
        if cache is not None:
            engram_cache = cache.setdefault("engram", {})
            attn_cache = cache.setdefault("attn", {})
        if self.engram is not None:
            x = x + self.engram(x, ids, engram_cache)
        x = x + self.attn(self.ln1(x), attn_cache, pos0)
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        engram_layers = tuple(cfg.engram_layers)
        blocks = []
        for i in range(cfg.n_layers):
            blocks.append(Block(cfg, i, i in engram_layers))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def set_engram_compression(self, compress):
        for b in self.blocks:
            if b.engram is not None:
                b.engram.compress.copy_(compress.to(b.engram.compress.device))

    def set_attention_backend(self, use_sdpa):
        for b in self.blocks:
            b.attn.use_sdpa = use_sdpa

    def forward(self, idx):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, idx)
        x = self.ln_f(x)
        return self.head(x)

    def step(self, idx, cache=None):
        # incremental forward for generation: prompt first, then one token at a time
        if cache is None:
            layers = []
            for _ in self.blocks:
                layers.append({})
            cache = {"pos": 0, "layers": layers}
        pos0 = cache["pos"]
        assert pos0 + idx.shape[1] <= self.cfg.max_seq_len, "generation exceeded the trained context"
        x = self.tok_emb(idx)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, idx, cache["layers"][i], pos0)
        cache["pos"] = pos0 + idx.shape[1]
        return self.head(self.ln_f(x)), cache
