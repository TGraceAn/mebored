import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
from transformer import SelfAttentionHead, MultiheadSelfAttention, ModelConfig, FeedForward, Block

"""Yeah this from Google's Music Transformer...
Since I don't really remember the whole paper...
I'm just trying the relative attention with skew method for now"""

class MusicTransformer_AttentionHead(nn.Module):
    """..."""

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

        #TODO: Write the rest from here

        #relative embedding
        self.E = nn.Embedding(config.block_size, head_size) # use for relative positional encoding

        #to here

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

    #TODO: Check skew algorithm
    def skew(self, x):
        B, T, T2 = x.size()
        assert T == T2, 'The input must be square from dim 1, and 2'
        x = F.pad(x, (1, 0)) #Pad T on the left with zeros
        x = x.view(B, T + 1, T)
        return x[:, 1:] # B, T, T