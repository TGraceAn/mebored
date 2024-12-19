import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
from transformer import SelfAttentionHead, MultiheadSelfAttention, ModelConfig, FeedForward, Block

"""Yeah this from Google's Music Transformer...
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

        self.E = nn.Embedding(config.block_size, head_size)

        #TODO: Write the rest here

    #Check skew
    def skew(self, x):
        B, T, T2 = x.size()
        assert T == T2, 'The input must be square from dim 1, and 2'
        x = F.pad(x, (1, 0)) #Pad T on the left with zeros
        x = x.view(B, T + 1, T)
        return x[:, 1:] # B, T, T