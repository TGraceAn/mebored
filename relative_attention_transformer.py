import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
from transformer import SelfAttentionHead, MultiheadSelfAttention, ModelConfig, FeedForward, Block

"""This from the paper "Self-Attention with Relative Position Representations"
But just the attention head and multihead attention block for now
Since I don't really remember the whole paper"""

class RelativeAttentionHead(nn.Module):
    """
    One head of Self-Attention with relative positional encoding
    Returns: output for one relative attention head
    """

    def __init__(self, config: ModelConfig, head_size: int):
        """
        Args:
            configs (ModelConfig): The hyperparameters of the model
            head_size (int): The size of the head
        """
        super().__init__()

        #Get the parameters
        self.config = config

        #Linear layers for key, query, value
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)

        # Attention mask (Optional)
        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size)))

        # # Optional for regularization default = 0
        # self.dropout = nn.Dropout(config.dropout)

        #relative embedding
        self.E = nn.Embedding(config.block_size, head_size) # use for relative positional encoding

        self.max_relative_len = config.block_size # max relative calculation
        #TODO: Write the rest here

    def forward(self, x):
        B, T, C = x.shape # B: batch_size, T: sequence_length, C: channels (n_embd)

        k = self.key(x)     # B, T, hs
        q = self.query(x)   # B, T, hs
        v = self.value(x)   # B, T, hs


        #TODO: Write the rest from here
        S_rel = ...
        # positional embedding
        relative_position = ...


        # q @ k transpose
        wei_content = q @ k.transpose(-2, -1) # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        #to here

    # def _get_relative_positions(self):
    #     # Generate relative position indices
    #     range_vec = torch.arange(seq_len)
    #     dist_mat = range_vec[None, :] - range_vec[:, None]
    #     dist_mat_clipped = dist_mat.clamp(-max_seq_len + 1, max_seq_len - 1)
    #     dist_mat_shifted = dist_mat_clipped + max_seq_len - 1
    #     return dist_mat_shifted