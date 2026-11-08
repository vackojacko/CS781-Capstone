# small Arch 2 for the ablations: GQA + RoPE, RMSNorm, SwiGLU, with switches to turn each one off
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LMConfig:
    def __init__(self, vocab_size, d_model=640, n_layers=12, n_heads=10, n_kv_heads=2, d_ff=1706,
                 max_seq_len=1024, dropout=0.0, pos_emb_type="rope", norm_type="rmsnorm", ffn_type="swiglu",
                 scale_emb=False):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pos_emb_type = pos_emb_type
        self.norm_type = norm_type
        self.ffn_type = ffn_type
        self.scale_emb = scale_emb


def build_rope_cache(max_seq_len, d_head, base=10000.0):
    half = d_head // 2
    freqs = 1.0 / (base ** (torch.arange(0, half).float() / half))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len, dropout=0.0, use_rope=True):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        cos, sin = build_rope_cache(max_seq_len, self.d_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


# sinusoidal positions from Arch 1 / A3
def sinusoidal_positional_encoding(max_len, d_model):
    pos_encoding = torch.zeros(max_len, d_model)
    for pos in range(max_len):
        for i in range(d_model):
            if i % 2 == 0:
                pos_encoding[pos, i] = math.sin(pos / 10000 ** (i / d_model))
            else:
                pos_encoding[pos, i] = math.cos(pos / 10000 ** ((i - 1) / d_model))
    return pos_encoding


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# GELU MLP from Arch 1
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff, bias=True), nn.GELU(), nn.Linear(d_ff, d_model, bias=True),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


def make_norm(cfg):
    if cfg.norm_type == "layernorm":
        return nn.LayerNorm(cfg.d_model)
    return RMSNorm(cfg.d_model)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = make_norm(cfg)
        use_rope = cfg.pos_emb_type == "rope"
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.max_seq_len, cfg.dropout,
                                        use_rope=use_rope)
        self.ln2 = make_norm(cfg)
        if cfg.ffn_type == "gelu":
            self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_emb_type == "sinusoidal":
            pe = sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model)
            self.register_buffer("pos_emb", pe, persistent=False)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = make_norm(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.cfg.scale_emb:
            x = x * math.sqrt(self.cfg.d_model)
        if self.cfg.pos_emb_type == "sinusoidal":
            x = x + self.pos_emb[:T]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
