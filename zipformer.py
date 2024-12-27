import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
from transformer import MultiheadSelfAttention, ModelConfig, FeedForward, Block, AttentionBlockOnly

"""Yeah, idk...."""

@dataclass
class ZipformerModelConfig(ModelConfig):
    ...


class MultiHeadAttentionWeight(nn.Module):
    """
    Calculate attention weight for multiple heads.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Args:
            config (ModelConfig): the hyperparameters of the model
        """
        super().__init__()
        hs = config.n_embd // config.n_head
        self.heads = nn.ModuleList([AttentionBlockOnly(config, hs) for _ in range(config.n_head)])

    def forward(self, x, mask: bool = False):
        """
        Args:
            x: Input tensor
            mask (bool: default = False): Apply mask
        Returns: output tensor
        """
        B, T, C = x.shape # (B, T, C) (C here is n_embed)

        # Feed through multiple heads and concat       
        if mask:
            out = torch.cat([h(x, mask=True) for h in self.heads], dim=-1) #(B, T, T*n_heads)
        else:
            out = torch.cat([h(x) for h in self.heads], dim=-1) #(B, T, T*n_heads)

        return out


class SA(nn.Module):
    """
    Calculate self-attention with the Multihead Attention Weight
    """

    def __init__(self, config: ModelConfig):
        """
        Args:
            config (ModelConfig): the hyperparameters of the model
        """
        self.config = config
        self.hs = config.n_embd // config.n_head
        self.value = nn.Linear(config.n_embd, config.n_head*self.hs, bias=False) #(B, T, C)
        self.proj = nn.Linear(config.n_head*self.hs, config.n_embd)

    def forward(self, x, wei):
        B, T, C = x.shape
        x = self.value(x) # (B, T, C) or (B, T, hs*num_head)
        # wei shape (B, T, T*num_head)

        wei = wei.view(B, T, T, self.config.n_head) # (B, T, T, num_head)
        x = x.view(B, T, self.config.n_head, self.hs) # (B, T, num_head, hs)

        out = torch.einsum('bttn,btnh->bthn', wei, x) # (B, T, T, num_head) efficient matmul (B, T, num_head, hs) -> (B, T, hs, num_head)
        out = out.reshape(B, T, -1)
        out = self.proj(out)

        return out # (B, T, n_embd)


class BiasNormFunction():
    """
    BiasNormFunction, this computes the BiasNorm
    In the context when BiasNorm is used in place of LayerNorm, this is use inplace of F.layernorm()
    """
    ...


class BiasNorm(nn.Module):
    """
    BiasNorm from the Zipformer paper
    (Use in place of LayerNorm)
    """

    def __init__(self):
        ...


class Bypass(nn.Module):
    """
    Bypass from the Zipformer paper
    As what I understand, it uses the input x from the original tensor and y which is after the last layer
    """

    def __init__(self):
        ...


class NonLinearAttention(nn.Module):
    """ One head of Non-Linear Attention:
        Returns: output for one non-linear attention head"""
    
    def __init__(self, config: ZipformerModelConfig):
        """
        Args:
            config (ZipformerModelConfig): the hyperparameters of the model
            w_a: attention weights
        """   
        # Get config
        self.config = config

        #TODO: write the rest here
        self.linear_scale = 3 * config.n_embd // 4

        self.linear_1 = nn.Linear(config.n_embd, self.linear_scale)
        self.linear_2 = nn.Linear(config.n_embd, self.linear_scale)
        self.linear_3 = nn.Linear(config.n_embd, self.linear_scale)

        self.tanh = nn.Tanh()

        self.last_linear = nn.Linear(self.linear_scale, config.n_embd)

    def forward(self, x, wei):
        # x.shape is (B, T, C)
        B, T, C = x.shape

        x_A, x_B, x_C = self.linear_1(x), self.linear_2(x), self.linear_3(x) #each is (B, T, (3/4)*C) or (B, T, hs*num_head*3/4)
        x_B = self.tanh(x_B) # check if this correct (B, T, (3/4)*C) or (B, T, hs*3/4*num_head)
        x_BC = x_B * x_C # use as value

        # wei shape (B, T, T*num_head)
        wei = wei.view(B, T, T, self.config.n_head) # (B, T, T, num_head)
        x_BC = x_BC.view(B, T, self.config.n_head, self.linear_scale) # (B, T, num_head, hs*3/4)
        out = torch.einsum('bttn,btnh->bthn', wei, x_BC) # (B, T, T, num_head) efficient matmul (B, T, num_head, hs*3/4) -> (B, T, 3/4*hs, num_head)
        out = out.reshape(B, T, -1) # (B, T, 3/4*hs*num_head)
        out = x_A * out 
        out = self.last_linear(out) # (B, T, C)
        
        return out


class ZipformerBlock(Block):
    """ Zipformer Block
        Returns: output for Zipformer"""
    def __init__(self, config: ZipformerModelConfig):
        """
        Args:
            config (ZipformerModelConfig): the hyperparameters of the model
        """   
        super().__init__()
        # Get config
        self.config = config

        # Model blocks
        #TODO: write the rest here

        self.mha = MultiHeadAttentionWeight(config)
        self.ffw_1 = FeedForward(...)
        self.nla = NonLinearAttention(config)
        self.sa_1 = SA(config)
        self.conv_1 = ...
        self.ffw_2 = FeedForward(...)
        self.bypass_1 = ...
        self.sa_2 = SA(config)
        self.conv_2 = ...
        self.ffw_3 = FeedForward(...)
        self.bias_norm = ... # Could be layer norm?
        self.bypass_2 = ...

    def forward(self, x):
        #TODO: write the rest here: Figure out the bypass and the biasnorm for coding 

        x_1, wei = self.ffw_1(x), self.mha(x, mask=False) #Change for tasks
        #After 1st block
        x_1 = x + x_1       # Residual connection
        x_11 = x_1          # Residual     
        x_1 = self.nla(x, wei)      
        #After 2nd block
        x_1 = x_1 + x_11    # Residual connection
        x_11 = x_1          # Residual   
        x_1 = self.sa_1(x_1, wei)
        #After 3rd block
        x_1 = x_1 + x_11    # Residual connection
        x_11 = x_1          # Residual
        x_1 = self.conv_1(x_1)
        #After 4th block
        x_1 = x_1 + x_11    # Residual connection
        x_11 = x_1          # Residual
        x_1 = self.ffw_2(x_1)
        #After 5th block
        x_1 = x_1 + x_11    # Residual connection
        x_1 = self.bypass_1(x_1)    # x do smth here
        #After 6th block
        x_11 = x_1          # Residual
        x_1 = self.sa_2(x_1, wei)
        #After 7th block
        x_1 = x_1 + x_11    # Residual connection
        x_11 = x_1          # Residual
        x_1 = self.conv_2(x_1)
        #After 8th block
        x_1 = x_1 + x_11    # Residual connection
        x_11 = x_1          # Residual
        x_1 = self.ffw_3(x_1)
        #After 9th block
        x_1 = x_1 + x_11    # Residual connection
        x_1 = self.bias_norm(x_1)
        #After 10 block
        out = self.bypass_2(x_1)    # x do smth here
        #After 11th block
        return out
        
#TODO: Check the code logic for the nla, after that can continute with the bypass and biasnorm part

