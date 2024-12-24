import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
from transformer import SelfAttentionHead, MultiheadSelfAttention, ModelConfig, FeedForward, Block

"""Yeah, idk...."""

@dataclass
class ZipformerModelConfig(ModelConfig):
    ...


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
    
    def __init__(self, config: ZipformerModelConfig, w_a):
        """
        Args:
            config (ZipformerModelConfig): the hyperparameters of the model
            w_a: attention weights
        """   
        # Get config
        self.config = config

        #TODO: write the rest here
        self.linear_scale = int((3/4)*config.n_embd)

        self.linear_1 = nn.Linear(config.n_embd, self.linear_scale)
        self.linear_2 = nn.Linear(config.n_embd, self.linear_scale)
        self.linear_3 = nn.Linear(config.n_embd, self.linear_scale)

        self.w_a = w_a #some attention weights

        self.last_linear = nn.Linear(self.linear_scale, config.n_embd)

    def forward(self, x):
        # x.shape is (B, T, C)
        x_1, x_2, x_3 = self.linear_1(x), self.linear_2(x), self.linear_3(x) #each is (B, T, (3/4)*C)
        x_1 = nn.Tanh(x_1) # check if this correct
        x_12 = x_1 * x_2 # pair-wise multiplication
        x_12 = torch.matmul(x_12, self.w_a)

        #TODO: write the rest here
        ...

        out = self.last_linear(...)

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

        self.mha = MultiheadSelfAttention(...)
        self.ffw_1 = FeedForward(...)
        self.nla = NonLinearAttention(ZipformerModelConfig, self.mha.weight)
        self.sa_1 = SelfAttentionHead(...)
        self.conv_1 = ...
        self.ffw_2 = FeedForward(...)
        self.bypass_1 = ...
        self.sa_2 = SelfAttentionHead(...)
        self.conv_2 = ...
        self.ffw_3 = FeedForward(...)
        self.bias_norm = ... # Could be layer norm?
        self.bypass_2 = ...

        # Weight sharing scheme from mha to the rest
        self.sa_1.weight = self.mha.weight
        self.sa_2.weight = self.mha.weight

    def forward(self, x):
        #TODO: write the rest here: Figure out the bypass and the biasnorm for coding 

        x_1, x_2 = self.ffw_1(x), self.mha(x)
        #After 1st block
        x_1 = x + x_1       # Residual connection
        x_11 = x_1          # Residual     
        x_1 = self.nla(x, x_2)      
        #After 2nd block
        x_1 = x_1 + x_11    # Residual connection
        x_11 = x_1          # Residual   
        x_1 = self.sa_1(x_1)    # x_2 do smth here
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
        x_1 = self.sa_2(x_1)    # x_2 do smth here
        #After 7th block
        x_1 = x_1 + x_11    # Residual connection
        x_11 = x_1          # Residual
        x_1 = self.conv_2(x_2)
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
        


