import sys
from abc import abstractmethod
import numpy as np


def broadcast_grad(grad, v):
    if grad.shape != v.shape:
        dims = [i for i, (d1, d2) in enumerate(zip(grad.shape, v.shape)) if d1 != d2]
        return np.sum(grad, axis=tuple(dims), keepdims=True)
    else:
        return grad


class Operator:
    def __init__(self, v):
        self.v = v

    @abstractmethod
    def backward(self, seed):
        return NotImplemented()


class BinaryOperator(Operator):
    def __init__(self, v, v1):
        super().__init__(v)
        self.v1 = v1

    @abstractmethod
    def _backward(self, seed): ...

    def backward(self, seed):
        v_grad, v1_grad = self._backward(seed)
        self.v.backward(broadcast_grad(v_grad, self.v))
        self.v1.backward(broadcast_grad(v1_grad, self.v1))


class Add(BinaryOperator):
    def _backward(self, seed):
        return seed, seed


class Sub(BinaryOperator):
    def _backward(self, seed):
        return seed, -seed


class Negative(Operator):
    def backward(self, seed):
        self.v.backward(-seed)


class Power(Operator):
    def __init__(self, v, power):
        super().__init__(v)
        self.power = power

    def backward(self, seed):
        if self.power == 0:
            grad = np.zeros_like(self.v)
        else:
            grad = self.power * self.v.value ** (self.power - 1)
        self.v.backward(seed * grad)


class Mult(BinaryOperator):
    def _backward(self, seed):
        return seed * self.v1.value, seed * self.v.value


class Sin(Operator):
    def backward(self, seed):
        self.v.backward(seed * np.cos(self.v.value))


class Cos(Operator):
    def backward(self, seed):
        self.v.backward(seed * -np.sin(self.v.value))


class MatMul(Operator):
    def __init__(self, v, v1):
        super().__init__(v)
        self.v1 = v1

    def backward(self, seed):
        self.v.backward(seed @ self.v1.value.T)
        self.v1.backward(self.v.value.T @ seed)


class Divide(BinaryOperator):
    def _backward(self, seed):
        return seed / self.v1.value, seed * -self.v.value / self.v1.value ** 2


class Transpose(Operator):
    def backward(self, seed):
        self.v.backward(seed.T)


class Index(Operator):
    def __init__(self, v, index):
        super().__init__(v)
        self.index = index

    def backward(self, seed):
        new_seed = np.zeros_like(self.v.value)
        # print(f'{seed.shape} {new_seed.shape} {self.index}')
        new_seed[self.index] = seed
        self.v.backward(new_seed)


class Concatenate:
    def __init__(self, dim, *values):
        self.dim = dim
        self.values = values

    def backward(self, seed):
        sections = np.cumsum([v.shape[self.dim] for v in self.values])[:-1]
        split_seed = np.split(seed, axis=self.dim, indices_or_sections=sections)
        for s, v in zip(split_seed, self.values):
            v.backward(s)


class Stack:
    def __init__(self, dim, *values):
        self.dim = dim
        self.values = values

    def backward(self, seed):
        unstacked_seed = np.unstack(seed, axis=self.dim)
        for v, s in zip(self.values, unstacked_seed):
            v.backward(s)


class Reshape(Operator):
    def backward(self, seed):
        self.v.backward(np.reshape(seed, self.v.value.shape))


class Sum(Operator):
    def __init__(self, v, dim, keepdims):
        super().__init__(v)
        self.dim = dim
        self.keepdims = keepdims

    def backward(self, seed):
        if self.dim is None or self.keepdims:
            grad = seed * np.ones_like(self.v.value)
        else:
            grad = np.repeat(np.expand_dims(seed, self.dim), self.v.shape[self.dim], self.dim)
        self.v.backward(grad)


class Sigmoid(Operator):
    def backward(self, seed):
        y = 1 / (1 + np.exp(-self.v.value))
        y *= (1 - y)
        self.v.backward(seed * y)


def _conv2d_f(x: np.ndarray, kernels: np.ndarray):
    result = []
    out_x_size = x.shape[-2] - kernels.shape[-2] + 1
    out_y_size = x.shape[-1] - kernels.shape[-1] + 1
    for i in range(out_x_size):
        for j in range(out_y_size):
            conv = x[..., i:i + kernels.shape[-2], j:j + kernels.shape[-1]] * kernels
            conv = np.sum(conv, axis=(-1, -2, -3))
            result.append(conv)
    result = np.concatenate(result, axis=-1)
    return result.reshape(kernels.shape[0], out_x_size, out_y_size)


class Conv2d(Operator):
    def __init__(self, v, k):
        super().__init__(v)
        self.k = k

    def get_seed_padding(self, seed):
        k_size = np.array(np.flip(self.k).shape[-2:])
        s_size = np.array(seed.shape)[-2:]
        v_shape = np.array(self.v.shape)[-2:]
        pad_x, pad_y = (k_size + v_shape - s_size) / 2
        return (int(pad_x), int(pad_x)), (int(pad_y), int(pad_y))

    def backward(self, seed):
        # print(f'S:{seed.shape} K: {self.k.shape} X: {self.v.shape}')
        grad_k = []
        # For each output channel, convolve with all input channels
        for s in seed:
            for x in self.v.value:
                grad_k.append(_conv2d_f(np.expand_dims(x, [0, 1]),
                                        np.expand_dims(s, [0, 1])))
        self.k.backward(np.array(grad_k).squeeze().reshape(self.k.shape))

        grad_c = [[] for _ in range(self.v.shape[0])]
        # For each pair of output and kernel, get the derivatives of each channel
        for s, k in zip(seed, self.k.value):
            for i, k_c in enumerate(k):
                k_c = np.flip(k_c)
                padded_seed = np.pad(s, self.get_seed_padding(s))
                grad_c[i].append(_conv2d_f(np.expand_dims(padded_seed, [0, 1]),
                                           np.expand_dims(k_c, [0, 1])))
        # Sum the derivatives channel wise
        grad_x = np.sum(np.array(grad_c), 1)
        grad_x = np.squeeze(grad_x)
        self.v.backward(np.array(grad_x, ndmin=len(self.v.shape)))


class Log(Operator):
    def backward(self, seed):
        self.v.backward(seed / (self.v.value + sys.float_info.epsilon))


class ReLU(Operator):
    def backward(self, seed):
        grad = np.ones_like(self.v.value)
        grad[self.v.value <= 0] = 0
        self.v.backward(seed * grad)


class Exp(Operator):
    def backward(self, seed):
        self.v.backward(seed * np.exp(self.v.value))


class Mean(Operator):
    def backward(self, seed):
        grad = np.ones_like(self.v) * 1 / self.v.value.size
        self.v.backward(seed * grad)
