import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
from transformer import ModelConfig

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
        self.E = nn.Embedding(2 * config.block_size - 1, head_size) # use for relative positional encoding

    def forward(self, x, mask: bool = False):
        """
        Args:
            x: Input tensor
            mask (bool: default = False): Apply mask
        Returns: output tensor
        """
        B, T, C = x.shape # B: batch_size, T: sequence_length, C: channels (n_embd)

        k = self.key(x)     # B, T, hs
        q = self.query(x)   # B, T, hs
        v = self.value(x)   # B, T, hs

        # Get the relative attention stuff
        R = self._get_rel_pos_embed(self.E) #relative position embedding
        # Q and R matrix multiplication operation
        rel_att = q.unsqueeze(2) @ R.transpose(-2, -1) # q from (B, T, hs) -> (B, T, 1, hs) @ (T, hs, T) -> (B, T, 1, T) # Trust me, it works, it's not R with full transposed, I tested
        S_rel = rel_att.squeeze(2) # (B, T, T)

        # q @ k transpose
        wei_content = q @ k.transpose(-2, -1) # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        
        #Add the 2 together and scale to get relative attention weight
        wei = (wei_content + S_rel) * k.shape[-1]**-0.5

        if mask:
            wei = wei.maskfill(self.mask[:T, :T] == 0, float('-inf'))

        # # Dropout
        # wei = self.dropout(wei)

        # Apply softmax
        wei = F.softmax(wei) # (B, T, T)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

    def _get_rel_pos_embed(self, E: nn.Embedding):
        """
        Args:
            E (nn.Embedding): The embedding layer, use for getting the embeddings
        Return:
            relative_position_embedding matrix
        """
        seq_len = self.config.block_size
        range_vec = torch.arange(seq_len)
        dist_mat = range_vec[None, :] - range_vec[:, None]
        dist_mat_clipped = dist_mat.clamp(-seq_len + 1, seq_len - 1)
        dist_mat_shifted = dist_mat_clipped + seq_len - 1
        dist_mat_embed = E(dist_mat_shifted)
        return dist_mat_embed


class MultiheadRelativeAttention(nn.Module):
    """ Multihead RelativeAttention Mechanism:
        Returns: output for multihead-relative-attention """
    def __init__(self, config: ModelConfig):
        """
        Args:
            config (ModelConfig): the hyperparameters of the model
        """
        super().__init__()
        hs = config.n_embd // config.n_head
        self.heads = nn.ModuleList([RelativeAttentionHead(config, hs) for _ in range(config.block_size)])

        #Last proj layer
        self.proj = nn.Linear(hs*config.n_head, config.n_head)

        # self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask):
        """
        Args:
            x: Input tensor
            mask (bool: default = False): Apply mask
        Returns: output tensor
        """
        B, T, C = x.shape # (B, T, C) (C here is n_embed)

        # Feed thourgh the heads and concat
        if mask:
            out = torch.cat([h(x, mask=True) for h in self.heads], dim=-1) #(B, T, hs*n_heads: should be num_embed)
        else: 
            out = torch.cat([h(x) for h in self.heads], dim=-1)

        #Projection Layer
        out = self.proj(out)

        return out

#TODO: fix the typing!!!!