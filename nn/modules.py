import math
from math import sqrt
from typing import Optional

import numpy as np
from autodiff.functions import conv2d, relu, layer_norm, masked_fill, softmax, reshape
from autodiff.variable import Variable, transpose


def _calculate_fan_in_fan_out(shape) -> tuple[int, int]:
    in_size, out_size = shape[1], shape[0]
    receptive_field = math.prod(shape[2:])
    return in_size * receptive_field, out_size * receptive_field


def xavier_init(*dim) -> Variable:
    fan_in, fan_out = _calculate_fan_in_fan_out(dim)
    std = math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return Variable(np.random.uniform(-a, a, dim), requires_grad=True)


def kaiming_init(size):
    a = 1 / math.sqrt(size[0])
    return Variable(np.random.uniform(-a, a, size), requires_grad=True)


def normal_init(size):
    return Variable(np.random.normal(size=size), requires_grad=True)


def attention(q: Variable, k: Variable, v: Variable, mask=None) -> Variable:
    d_k = Variable([sqrt(q.shape[-1])])
    y = q @ transpose(k, -2, -1) / d_k
    if mask is not None:
        y = masked_fill(y, mask, -10000)
    y = softmax(y, dim=-1)
    return y @ v


class Module():
    def __init__(self):
        super().__init__()
        self.training = True

    def forward(self, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        params = []
        for a in vars(self).values():
            if isinstance(a, Variable) and a.requires_grad:
                params.append(a)
            elif isinstance(a, Module):
                params += a.parameters()
        return params

    def set_training(self, train: bool):
        for a in vars(self).values():
            if isinstance(a, Module):
                a.set_training(train)


class LayerNorm(Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.w = Variable(np.ones(normalized_shape), requires_grad=True)
        self.b = Variable(np.zeros(normalized_shape), requires_grad=True)

    def forward(self, v):
        return layer_norm(v, self.w, self.b)


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = normal_init((num_embeddings, embedding_dim))

    def forward(self, idx: Variable):
        return self.weight[idx.value]


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def __iter__(self):
        return self.modules.__iter__()

    def forward(self, x):
        y = x
        for m in self.modules:
            y = m(y)
        return y

    def parameters(self):
        params = []
        for m in self.modules:
            params += m.parameters()
        return params


class ReLU(Module):
    def forward(self, x):
        return relu(x)


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w, self.b = self.initi_params((out_features, in_features), (out_features,))

    def initi_params(self, weight_shape, bias_shape):
        fan_in, _ = _calculate_fan_in_fan_out(weight_shape)
        gain = 2 / (1 + 5)
        w = Variable(sqrt(gain * 3 / fan_in) * np.ones(weight_shape), requires_grad=True)

        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        b = Variable(np.random.uniform(-bound, bound, bias_shape), requires_grad=True)

        return w, b

    def forward(self, x) -> Variable:
        y = x @ self.w.T + self.b
        return y


class Convolution2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple):
        super().__init__()
        k_size = (in_channels,) + kernel_size
        self.kernels = [kaiming_init(k_size).value for _ in range(out_channels)]
        self.kernels = Variable(self.kernels, requires_grad=True)
        self.b = kaiming_init((out_channels, 1, 1))

    def forward(self, x):
        return conv2d(x, self.kernels) + self.b


class MultiHeadAttention(Module):
    def __init__(self, in_dim: int, embed_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.w_k = xavier_init(in_dim, self.embed_dim)
        self.w_q = xavier_init(in_dim, self.embed_dim)
        self.w_v = xavier_init(in_dim, self.embed_dim)
        self.w_o = xavier_init(self.embed_dim, in_dim)

    def forward(self, q: Variable, k: Variable, v: Variable, mask: Optional[np.ndarray] = None) -> Variable:
        # Project
        q_proj = q @ self.w_q
        k_proj = k @ self.w_k
        v_proj = v @ self.w_v

        # Split into multiple heads
        batch_size = q.shape[0]
        n_samples = q.shape[-2]
        head_shape = (batch_size, self.n_heads, n_samples, self.embed_dim // self.n_heads)
        q_head = reshape(transpose(q_proj, -2, -1), head_shape)
        k_head = reshape(transpose(k_proj, -2, -1), head_shape)
        v_head = reshape(transpose(v_proj, -2, -1), head_shape)

        attn_out = attention(q_head, k_head, v_head, mask)

        # # Concatenate
        attn_out = transpose(attn_out, -3, -2)
        attn_out = reshape(attn_out, (batch_size, n_samples, self.embed_dim))

        # Reproject
        return attn_out @ self.w_o
