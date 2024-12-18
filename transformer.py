import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod

"""
This is me when I'm too bored
"""

@dataclass
class ModelConfig:
    """
    Config for the Transfomer model, modify depend on use:
    Hyperparams:
        block_size (int): context length
        n_layer (int): number of hidden layers/num_block (how deep the model is)
        n_head (int): number of heads (how many heads will be working)
        n_embd (int): embedding size (how long the vector for a word) (= n_head * head_size)
        vocab_size (int): number of vocab in the tokenizer
        dropout (Optional[int]): default 0, percentage of dropouts
    """
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int
    dropout: Optional[int] = 0


class SelfAttentionHead(nn.Module):
    """ One head of Self-Attention:
        Returns: output for one attention head"""

    def __init__(self, config: ModelConfig, head_size: int):
        """
        Args:
            config (ModelConfig): the hyperparameters of the model
            head_size (int): how big a head is
        """            

        super().__init__()

        # Get config
        self.config = config

        # Linear layer for key, query, value
        self.key = nn.Linear(self.config.n_embd, head_size, bias = False) # map from n_embd -> hs
        self.query = nn.Linear(self.config.n_embd, head_size, bias = False)
        self.value = nn.Linear(self.config.n_embd, head_size, bias = False)

        # Attention mask (Optional)
        self.register_buffer('mask', torch.tril(torch.ones(self.config.block_size, self.config.block_size)))        
        
        # # Optional for regularization default = 0
        # self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x, mask: bool = False):
        """
        Args:
            x: Input tensor
            mask (bool: default = False): Apply mask
        Returns: output tensor
        """

        B, T, C = x.shape # B: batch_size, T: sequence_length, C: channels (n_embd)
        
        # Get keys, query, value
        k = self.key(x)     # B, T, hs
        q = self.query(x)   # B, T, hs
        v = self.value(x)   # B, T, hs

        # Calculate the attention weights
        wei = q @ k.transpose(-2, -1) * k.shape(-1)**-0.5 #scaled dot product (B, T, hs) @ (B, hs, T) -> (B, T, T)
        
        if mask:
            # Mask (Optional)
            wei = wei.maskfill(self.mask[:T, :T] == 0, float('-inf')) # (B, T, T)
        
        # # Dropout
        # wei = self.dropout(wei)

        # Apply softmax
        wei = F.softmax(wei) # (B, T, T)

        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiheadSelfAttention(nn.Module):
    """ Multihead Self-Attention Mechanism:
        Returns: output for multihead-attention """

    def __init__(self, config: ModelConfig):
        """
        Args:
            config (ModelConfig): the hyperparameters of the model
        """
        super().__init__()
        #Get model config
        self.config = config

        self.hs = self.config.n_embd // self.config.n_head
        self.heads = nn.ModuleList([SelfAttentionHead(self.config, self.hs) for _ in range(self.config.n_head)]) #get the number of heads to be calculated in parallel

        # Last linear layer
        # Normally hs*heads = n_embed
        self.proj = nn.Linear(self.hs*self.config.n_head, self.config.n_embd)

        # # Optional for regularization default = 0
        # self.dropout = nn.Dropout(self.config.dropout)

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
            out = torch.cat([h(x, mask=True) for h in self.heads], dim=-1) #(B, T, hs*n_heads)
        else:
            out = torch.cat([h(x) for h in self.heads], dim=-1) #(B, T, hs*n_heads)

        # Last linear layer
        out = self.proj(x) # (B, T, C) (C here is n_embed)

        # # Dropout
        # out = self.dropout(out)

        return out


class CrossAttentionHead(nn.Module):
    """ One head of Cross-Attention:
        Returns: output for one attention head"""

    def __init__(self, config: ModelConfig, head_size: int):
        """
        Args:
            config (ModelConfig): the hyperparameters of the model
            head_size (int): how big a head is
        """            

        super().__init__()

        # Get config
        self.config = config

        # Linear layer for key, query, value
        self.key = nn.Linear(self.config.n_embd, head_size, bias = False) # map from n_embd -> hs
        self.query = nn.Linear(self.config.n_embd, head_size, bias = False)
        self.value = nn.Linear(self.config.n_embd, head_size, bias = False)

        # Attention mask (Optional)
        self.register_buffer('mask', torch.tril(torch.ones(self.config.block_size, self.config.block_size)))        
        
        # # Optional for regularization default = 0
        # self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x, encode_out, mask: bool = False):
        B, T, C = x.shape # B: batch_size, T: sequence_length, C: channels (n_embd)
        """
        Args:
            x: Input tensor
            encode_out: output of the encoder
            mask (bool: default = False): Apply mask
        Returns: output tensor
        """
        # Get keys, query, value
        k = self.key(encode_out)        # B, T, hs
        q = self.query(x)               # B, T, hs
        v = self.value(encode_out)      # B, T, hs

        # Calculate the attention weights
        wei = q @ k.transpose(-2, -1) * k.shape(-1)**-0.5 #scaled dot product (B, T, hs) @ (B, hs, T) -> (B, T, T)
        
        if mask:
            # Mask (Optional)
            wei = wei.maskfill(self.mask[:T, :T] == 0, float('-inf')) # (B, T, T)
        
        # # Dropout
        # wei = self.dropout(wei)

        # Apply softmax
        wei = F.softmax(wei) # (B, T, T)

        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiheadCrossAttention(nn.Module):
    """ Multihead Cross-Attention Mechanism:
        Returns: output for multihead-attention """

    def __init__(self, config: ModelConfig):
        """
        Args:
            config (ModelConfig): the hyperparameters of the model
        """
        super().__init__()
        #Get model config
        self.config = config

        self.hs = self.config.n_embd // self.config.n_head
        self.heads = nn.ModuleList([CrossAttentionHead(self.config, self.hs) for _ in range(self.config.n_head)]) #get the number of heads to be calculated in parallel

        # Last linear layer
        # Normally hs*heads = n_embed
        self.proj = nn.Linear(self.hs*self.config.n_head, self.config.n_embd)

        # # Optional for regularization default = 0
        # self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x, encode_out, mask: bool = False):
        """
        Args:
            x: Input tensor
            encode_out: output of the encoder
            mask (bool: default = False): Apply mask
        Returns: output tensor
        """

        B, T, C = x.shape # (B, T, C) (C here is n_embed)

        # Feed through multiple heads and concat        
        if mask:
            out = torch.cat([h(x, encode_out, mask=True) for h in self.heads], dim=-1) #(B, T, hs*n_heads)
        else:
            out = torch.cat([h(x, encode_out) for h in self.heads], dim=-1) #(B, T, hs*n_heads)

        # Last linear layer
        out = self.proj(x) # (B, T, C) (C here is n_embed)

        # # Dropout
        # out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    """FeedForward layer in the transformer (In GPT-2 it's mlp and uses GELU instead of ReLU)      
        Returns: output for the feedforward layer
    """
    def __init__(self, config: ModelConfig):
        """
        Args:
            config (ModelConfig): the hyperparameters of the model
        """
        super().__init__()

        # Get the config
        self.config = config

        # Network for the Feedforward layer
        self.net = nn.Sequencial(
            nn.Linear(self.config.n_embd, 4 * self.config.n_embd), #Expand the hidden size
            nn.ReLU(),
            nn.Linear(4*self.config.n_embd, self.config.n_embd)
            # , nn.Dropout(self.config.dropout) 
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor
        Returns: Output of the FeedForward layer
        """

        # x shape (B, T, C) (C here is n_embed)
        return self.net(x)


class Block(nn.Module, ABC):
    """Interface for writing encoder & decoder blocks"""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Each block has different implementation"""
        ...


class EncoderBlock(Block):
    """A single Encoder block
        Returns: output for one encoder block
        """
    def __init__(self, config: ModelConfig):
        """
        Args:
            config (ModelConfig): the hyperparameters of the model
        """
        super().__init__()

        self.config = config
        self.attention = MultiheadSelfAttention(self.config)
        self.ln_1 = nn.LayerNorm(self.config.n_embd)
        self.ln_2 = nn.LayerNorm(self.config.n_embd)
        self.ffw = FeedForward(self.config)

    def forward(self, x):
        """
        Args:
            x: Input tensor
        Returns: Encoder output 
        """
        # x shape (B, T, C) (C here is n_embed)
        # this follows the original paper
        x = self.ln_1(x + self.attention(x))
        x = self.ln_2(x + self.ffw(x))

        # # this follows the recent studies
        # x = x + self.attention(self.ln_1(x))
        # x = x + self.ffw(self.ln_2(x))
        return x


class DecoderBlock(nn.Module):
    """A single Decoder block
        Returns: output for one decoder block
        """
    def __init__(self, config: ModelConfig):
        """
        Args:
            config (ModelConfig): the hyperparameters of the model
        """
        super().__init__()
        
        # Get the config
        self.config = config
        
        # Blocks in this block
        self.self_attention = MultiheadSelfAttention(self.config) #remember to set mask=True in forward
        self.cross_attention = MultiheadCrossAttention(self.config)
        self.ffw = FeedForward(self.config)
        self.ln_1 = nn.LayerNorm(self.config.n_embd)
        self.ln_2 = nn.LayerNorm(self.config.n_embd)
        self.ln_3 = nn.LayerNorm(self.config.n_embd)

    def forward(self, x, encode_out):
        """
        Args:
            x: Input tensor
            encode_out: Encoder output
        Returns: Decoder output 
        """
        # x shape (B, T, C) (C here is n_embed)
        # this follows the original paper
        x = self.ln_1(x + self.self_attention(x, mask=True))
        x = self.ln_2(x + self.cross_attention(x, encode_out))
        x = self.ln_3(x + self.ffw(x))

        # # this follows the recent studies
        # x = x + self.self_attention(self.ln_1(x))
        # x = x + self.cross_attention(self.ln_2(x))
        # x = x + self.ffw(self.ln_3(x))

        return x


class Encoder(nn.Module):
    """ Encoder with N layers
        Returns: encoder output for the decoder
        """
    def __init__(self, config: ModelConfig):
        """
        Args:
            config (ModelConfig): the hyperparameters of the model
        """
        super().__init__()

        # Get the config
        self.config = config

        self.blocks = nn.ModuleList([EncoderBlock(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd) 
        
    def forward(self, x):
        """
        Args:
            x: Input tensor
        Returns: Output of N encoder blocks
        """

        for block in self.blocks:
            x = block(x)
        # x = self.ln_f(x) # This is probably unnecessary cause the output of the encoder already has layer_norm for the vanilla case
        return x


class Decoder(nn.Module):
    """ Decoder with N layers
        Returns: output for calculatimg probabilities
        """
    def __init__(self, config: ModelConfig):
        """
        Args:
            config (ModelConfig): the hyperparameters of the model
        """
        super().__init__()

        # Get the config
        self.config = config

        self.blocks = nn.ModuleList([DecoderBlock(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd) 
        
    def forward(self, x, encode_out):
        """
        Args:
            x: Input tensor
            encode_out: Encoder output
        Returns: Output of N decoder blocks
        """
        for block in self.blocks:
            x = block(x, encode_out)
        # x = self.ln_f(x) # This is probably unnecessary cause the output of the encoder already has layer_norm for the vanilla case
        return x


#Valilla Cross Attention Transformer
class VanillaTransfomer(nn.Module):
    """Vanilla Transformer.
        Returns: output probabilities
        """

    def __init__(self, config: ModelConfig):
        """
        Args:
            config (ModelConfig): the hyperparameters of the model
        """
        super().__init__()

        # Get the configs
        self.config = config

        # Get the architecture
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

        # Embeddings: I'm still not sure if we're suppose to split it like this or use the same positional and token embedding for both
        self.src_wte = nn.Embedding(self.config.vocab_size, self.config.n_embd) #Token embedding
        self.src_wpe = nn.Embedding(self.config.block_size, self.config.n_embd) #Positional embedding
        self.tgt_wte = nn.Embedding(self.config.vocab_size, self.config.n_embd) #Token embedding
        self.tgt_wpe = nn.Embedding(self.config.block_size, self.config.n_embd) #Positional embedding
        
        # Output projection layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Init weights
        self.apply(self._init_weights)

        # # Weight sharing scheme
        # ...

    # This helps the model converge faster
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            #Residule connection control
            std *= (2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) #init bias weights to become 0
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)

    def forward(self, src, tgt):
        """
        Args:
            src (torch.Tensor): Source sequence of shape (B, T_src)
            tgt (torch.Tensor): Target sequence of shape (B, T_tgt)
        Returns:
            logits (torch.Tensor): Output logits of shape (B, T_tgt, vocab_size)
        """
        B_1, T_src = src.shape
        B_2, T_tgt = tgt.shape

        assert B_1 == B_2, 'make sure that the input and output have the same batch size'

        # Positional vectors
        src_pos = torch.arange(0, T_src, dtype = torch.long, device = src.device)
        tgt_pos = torch.arange(0, T_tgt, dtype = torch.long, device = tgt.device)

        # Token embedding
        src_tok = self.src_wte(src)
        tgt_tok = self.tgt_wte(tgt)

        # Inputs for the encode and decode
        src_x = src_pos + src_tok
        tgt_x = tgt_pos + tgt_tok

        encode_out = self.encoder(src_x)
        decode_out = self.decoder(tgt_x, encode_out)

        logits = self.lm_head(decode_out)
        return logits
    
#TODO: fix the typing!!!!