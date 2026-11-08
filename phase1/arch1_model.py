# Arch 1: GPT-2 style decoder (sinusoidal positions, pre-norm LayerNorm, GELU MLP, multi-head attention)
import math
import torch
import torch.nn as nn


class LMConfig:
    def __init__(self, vocab_size, d_model=512, n_layers=4, n_heads=4, n_kv_heads=4, d_ff=1024,
                 max_seq_len=256, dropout=0.1, pos_emb_type="sinusoidal", attention_scale=None, norm_first=False):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pos_emb_type = pos_emb_type
        self.attention_scale = attention_scale
        self.norm_first = norm_first


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.0, attention_scale=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if attention_scale is None:
            self.attention_scale = 1.0 / math.sqrt(self.d_head)
        else:
            self.attention_scale = float(attention_scale)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer("causal_mask", torch.tril(torch.ones(max_seq_len, max_seq_len)), persistent=False)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)
        v = self.v_proj(x)
        k = self.k_proj(x)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1) * self.attention_scale)
        scores = scores.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))

        A = torch.softmax(scores, dim=-1)
        A = self.attn_dropout(A)
        out = A @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


def sinusoidal_positional_encoding(max_len, d_model):
    pos_encoding = torch.zeros(max_len, d_model)
    for pos in range(max_len):
        for i in range(d_model):
            if i % 2 == 0:
                pos_encoding[pos, i] = math.sin(pos / 10000 ** (i / d_model))
            else:
                pos_encoding[pos, i] = math.cos(pos / 10000 ** ((i - 1) / d_model))
    return pos_encoding


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff, bias=True), nn.GELU(), nn.Linear(d_ff, d_model, bias=True), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.max_seq_len, cfg.dropout,
                                        attention_scale=cfg.attention_scale)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm_first = cfg.norm_first

    def forward(self, x):
        if self.norm_first is True:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        return x


class DecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if self.cfg.pos_emb_type == "sinusoidal":
            pe = sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model)
            self.register_buffer("pos_emb", pe, persistent=False)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx):
        B, T = idx.shape
        if self.cfg.pos_emb_type == "sinusoidal":
            x = self.tok_emb(idx) + self.pos_emb[:T]
        else:
            x = self.tok_emb(idx)
        for h in self.blocks:
            x = h(x)
        x = self.ln_f(x)
        x = self.head(x)
        return x
