# model/decoder.py — Decoder block + Decoder stack

import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.encoder import FeedForward


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention  = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff              = FeedForward(d_model, d_ff, dropout)
        self.norm1           = nn.LayerNorm(d_model)
        self.norm2           = nn.LayerNorm(d_model)
        self.norm3           = nn.LayerNorm(d_model)
        self.dropout         = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        # 1. masked self-attention (can't look at future tokens)
        self_attn = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn))

        # 2. cross-attention (attend to encoder output)
        cross_attn = self.cross_attention(x, encoder_out, encoder_out, src_mask)
        x = self.norm2(x + self.dropout(cross_attn))

        # 3. feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x  # (batch, tgt_seq_len, d_model)


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        return self.norm(x)  # (batch, tgt_seq_len, d_model)