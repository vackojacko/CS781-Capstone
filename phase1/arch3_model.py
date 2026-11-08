# Arch 3: encoder-decoder Transformer with RoPE, RMSNorm, SwiGLU and cross-attention
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LMConfig:
    def __init__(self, vocab_size, d_model=640, n_enc_layers=10, n_dec_layers=4, n_heads=10,
                 d_ff=1712, max_enc_len=704, max_dec_len=320, dropout=0.0):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.dropout = dropout


def build_rope_cache(max_seq_len, d_head, base=10000.0):
    assert d_head % 2 == 0, "RoPE requires an even head dimension"
    inverse_frequencies = torch.zeros(d_head // 2)
    for i in range(0, d_head // 2, 1):
        inverse_frequencies[i] = (1 / (base ** (2 * i / d_head)))

    phi_values = torch.zeros(max_seq_len, d_head // 2)
    positions = [i for i in range(0, max_seq_len)]
    for i in range(len(positions)):
        for j in range(d_head // 2):
            phi_values[i, j] = positions[i] * inverse_frequencies[j]

    # channel i and channel i + d_head/2 share the same angle
    angles = torch.cat([phi_values, phi_values], dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x):
    last_dim = x.shape[-1]
    half = last_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    ans = torch.cat([-x2, x1], dim=-1)
    return ans


def apply_rope(x, cos, sin):
    T, d_head = cos.shape
    cos = cos.view(1, 1, T, d_head)
    sin = sin.view(1, 1, T, d_head)
    x_rotated = x * cos + rotate_half(x) * sin
    return x_rotated


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len, dropout=0.0, use_rope=False, attention_scale=None):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.use_rope = use_rope
        if attention_scale is None:
            self.attention_scale = 1.0 / math.sqrt(self.d_head)
        else:
            self.attention_scale = float(attention_scale)

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

        if use_rope:
            cos, sin = build_rope_cache(max_seq_len, self.d_head)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
            k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) * self.attention_scale
        scores = scores.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
        probs = torch.softmax(scores, dim=-1)

        probs = self.attn_dropout(probs)
        out = probs @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class SelfAttention(nn.Module):
    # encoder attention: same as CausalSelfAttention but bidirectional, with a key-padding mask
    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len, dropout=0.0, use_rope=False, attention_scale=None):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.use_rope = use_rope
        if attention_scale is None:
            self.attention_scale = 1.0 / math.sqrt(self.d_head)
        else:
            self.attention_scale = float(attention_scale)

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if use_rope:
            cos, sin = build_rope_cache(max_seq_len, self.d_head)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, key_padding_mask=None):
        # key_padding_mask: (B, T) bool, True = real token
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
            k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) * self.attention_scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float('-inf'))
        probs = torch.softmax(scores, dim=-1)

        probs = self.attn_dropout(probs)
        out = probs @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class CrossAttention(nn.Module):
    # decoder queries attend to encoder keys/values, no RoPE
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attention_scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, enc_mask=None):
        B, T_dec, C = x.shape
        T_enc = enc_output.shape[1]

        q = self.q_proj(x)
        k = self.k_proj(enc_output)
        v = self.v_proj(enc_output)

        q = q.view(B, T_dec, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T_enc, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T_enc, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.attention_scale
        if enc_mask is not None:
            scores = scores.masked_fill(~enc_mask[:, None, None, :], float('-inf'))
        A = torch.softmax(scores, dim=-1)
        A = self.attn_dropout(A)
        out = A @ v

        out = out.transpose(1, 2).contiguous().view(B, T_dec, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # computed in fp32 so x*x cannot overflow under fp16 autocast
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = SelfAttention(cfg.d_model, cfg.n_heads, cfg.n_heads, cfg.max_enc_len,
                                  cfg.dropout, use_rope=True)
        self.ln2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x, enc_mask=None):
        x = x + self.attn(self.ln1(x), enc_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.self_attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.n_heads, cfg.max_dec_len,
                                             cfg.dropout, use_rope=True)
        self.ln2 = RMSNorm(cfg.d_model)
        self.cross_attn = CrossAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln3 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x, enc_output, enc_mask=None):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), enc_output, enc_mask)
        x = x + self.ffn(self.ln3(x))
        return x


class EncoderDecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)  # shared by encoder and decoder
        self.enc_blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_enc_layers)])
        self.enc_norm = RMSNorm(cfg.d_model)
        self.dec_blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_dec_layers)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def encode(self, enc_input, enc_mask=None):
        x = self.tok_emb(enc_input)
        for block in self.enc_blocks:
            x = block(x, enc_mask)
        return self.enc_norm(x)

    def decode(self, dec_input, enc_output, enc_mask=None):
        x = self.tok_emb(dec_input)
        for block in self.dec_blocks:
            x = block(x, enc_output, enc_mask)
        x = self.ln_f(x)
        return self.head(x)

    def forward(self, enc_input, dec_input, enc_mask=None):
        enc_output = self.encode(enc_input, enc_mask)
        return self.decode(dec_input, enc_output, enc_mask)
