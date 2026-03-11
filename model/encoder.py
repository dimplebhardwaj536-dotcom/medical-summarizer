# model/encoder.py — Encoder block + Encoder stack

import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention  = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff         = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # self-attention + residual
        attn_out = self.attention(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # feed-forward + residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x  # (batch, seq_len, d_model)


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)  # (batch, seq_len, d_model)