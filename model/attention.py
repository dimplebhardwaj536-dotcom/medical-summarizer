# model/attention.py — Multi-Head Attention from scratch

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads  # dimension per head

        # single linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V: (batch, heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, V)  # (batch, heads, seq_len, d_k)

    def split_heads(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, heads, seq_len, d_k)

    def combine_heads(self, x):
        # x: (batch, heads, seq_len, d_k)
        batch, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)  # (batch, seq_len, d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))

        return output  # (batch, seq_len, d_model)