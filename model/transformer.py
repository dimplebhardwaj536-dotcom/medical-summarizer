# model/transformer.py — Full Transformer model

import torch
import torch.nn as nn
import math
from model.encoder import Encoder
from model.decoder import Decoder
from config import config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # build positional encoding matrix once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)  # not a parameter, but saved with model

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        # embeddings
        self.src_embedding = nn.Embedding(config.VOCAB_SIZE, config.D_MODEL, padding_idx=config.PAD_TOKEN_ID)
        self.tgt_embedding = nn.Embedding(config.VOCAB_SIZE, config.D_MODEL, padding_idx=config.PAD_TOKEN_ID)

        # positional encoding
        self.src_pos = PositionalEncoding(config.D_MODEL, config.MAX_INPUT_LEN,  config.DROPOUT)
        self.tgt_pos = PositionalEncoding(config.D_MODEL, config.MAX_TARGET_LEN, config.DROPOUT)

        # encoder + decoder
        self.encoder = Encoder(config.D_MODEL, config.N_HEADS, config.D_FF, config.N_ENCODER_LAYERS, config.DROPOUT)
        self.decoder = Decoder(config.D_MODEL, config.N_HEADS, config.D_FF, config.N_DECODER_LAYERS, config.DROPOUT)

        # final projection to vocab
        self.output_layer = nn.Linear(config.D_MODEL, config.VOCAB_SIZE)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src_ids):
        # mask padding tokens: (batch, 1, 1, seq_len)
        return (src_ids != config.PAD_TOKEN_ID).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt_ids):
        batch, tgt_len = tgt_ids.size()

        # mask padding
        pad_mask = (tgt_ids != config.PAD_TOKEN_ID).unsqueeze(1).unsqueeze(2)

        # causal mask — can't look at future tokens
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt_ids.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        return pad_mask & causal_mask  # (batch, 1, tgt_len, tgt_len)

    def forward(self, src_ids, tgt_ids, src_mask=None):
        # masks
        if src_mask is None:
            src_mask = self.make_src_mask(src_ids)
        tgt_mask = self.make_tgt_mask(tgt_ids)

        # embeddings + positional encoding
        src = self.src_pos(self.src_embedding(src_ids))
        tgt = self.tgt_pos(self.tgt_embedding(tgt_ids))

        # encode + decode
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(tgt, encoder_out, src_mask, tgt_mask)

        # project to vocab size
        logits = self.output_layer(decoder_out)  # (batch, tgt_len, vocab_size)

        return logits