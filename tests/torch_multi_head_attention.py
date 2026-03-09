import math
import torch
from torch import nn, Tensor
from torch.nn import init, Parameter


def init_w(*dim):
    w = torch.empty(dim, dtype=torch.float)
    return init.xavier_uniform_(w)


def attention(q: Tensor, k: Tensor, v: Tensor, mask=None):
    d_k = q.shape[-1]
    y = q @ torch.transpose(k, -2, -1) / math.sqrt(d_k)
    if mask is not None:
        y = torch.masked_fill(y, mask, -10000)
    y = torch.softmax(y, dim=-1)
    return y @ v


class TorchMultiHeadAttention(nn.Module):
    """
    Implements a multi-head attention layer in pytorch to test against the numpy implementation.
    This version uses separate q, k, v and output weights.

    Args:
        in_dim (int): Input dimension.
        embed_dim (int): Embedding dimension.
        n_heads (int): Number of attention heads.
    """

    def __init__(self, in_dim: int, embed_dim: int, n_heads: int):
        super().__init__()

        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.w_k = Parameter(init_w(in_dim, self.embed_dim))
        self.w_q = Parameter(init_w(in_dim, self.embed_dim))
        self.w_v = Parameter(init_w(in_dim, self.embed_dim))
        self.w_o = Parameter(init_w(self.embed_dim, in_dim))

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None):
        """
         Computes the multi-head attention output.

         Args:
             q (Tensor): Query tensor of shape (batch_size, n_samples, in_dim).
             k (Tensor): Key tensor of shape (batch_size, n_samples, in_dim).
             v (Tensor): Value tensor of shape (batch_size, n_samples, in_dim).
             mask (Tensor, optional): Attention mask tensor.

         Returns:
             Tensor: The attention output tensor of shape (batch_size, n_samples, in_dim).
         """
        # Project
        q = q @ self.w_q
        k = k @ self.w_k
        v = v @ self.w_v

        # Split into multiple heads
        batch_size = q.shape[0]
        n_samples = q.shape[-2]
        q = q.transpose(-2, -1).reshape((batch_size, self.n_heads, n_samples, self.embed_dim // self.n_heads))
        k = k.transpose(-2, -1).reshape((batch_size, self.n_heads, n_samples, self.embed_dim // self.n_heads))
        v = v.transpose(-2, -1).reshape((batch_size, self.n_heads, n_samples, self.embed_dim // self.n_heads))

        attn_out = attention(q, k, v, mask)

        # Concatenate
        attn_out = attn_out.transpose(-3, -2).reshape((batch_size, n_samples, self.embed_dim))
        # Reproject
        attn_out = attn_out @ self.w_o
        return attn_out
