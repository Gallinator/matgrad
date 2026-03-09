import sys
from abc import abstractmethod
import numpy as np


def unbatch_grad(v, grad):
    # Batch dimensions
    if len(grad.shape) > len(v.shape):
        sum_dims = len(grad.shape) - len(v.shape)
        grad = np.sum(grad, axis=tuple(range(sum_dims)))
    return grad


def broadcast_grad(grad, v):
    if grad.shape != v.shape:
        grad = unbatch_grad(v, grad)
        dims = [i for i, (d1, d2) in enumerate(zip(grad.shape, v.shape)) if d1 != d2]
        return np.sum(grad, axis=tuple(dims), keepdims=True)
    else:
        return grad


class Operator():
    def __init__(self, *inputs):
        self.inputs = inputs

    @abstractmethod
    def backward(self, seed): ...


class UnaryOperator(Operator):
    def __init__(self, input):
        super().__init__()
        self.inputs = [input]

    @property
    def v(self):
        return self.inputs[0]

    @abstractmethod
    def _backward(self, seed): ...

    def backward(self, seed):
        self.v.accumulate(self._backward(seed))


class BinaryOperator(Operator):
    @property
    def v(self):
        return self.inputs[0]

    @property
    def v1(self):
        return self.inputs[1]

    @abstractmethod
    def _backward(self, seed): ...

    def backward(self, seed):
        v_grad, v1_grad = self._backward(seed)
        self.v.accumulate(broadcast_grad(v_grad, self.v))
        self.v1.accumulate(broadcast_grad(v1_grad, self.v1))


class Add(BinaryOperator):
    def _backward(self, seed):
        return seed, seed


class Sub(BinaryOperator):
    def _backward(self, seed):
        return seed, -seed


class Negative(UnaryOperator):
    def _backward(self, seed):
        return -seed


class Power(UnaryOperator):
    def __init__(self, input, power):
        super().__init__(input)
        self.power = power

    def _backward(self, seed):
        if self.power == 0:
            grad = np.zeros_like(self.v)
        else:
            grad = self.power * self.v.value ** (self.power - 1)
        return seed * grad


class Mult(BinaryOperator):
    def _backward(self, seed):
        return seed * self.v1.value, seed * self.v.value


class Sin(UnaryOperator):
    def _backward(self, seed):
        return seed * np.cos(self.v.value)


class Cos(UnaryOperator):
    def _backward(self, seed):
        return seed * -np.sin(self.v.value)


class MatMul(BinaryOperator):
    def _backward(self, seed):
        v_grad = seed @ np.swapaxes(self.v1.value, -1, -2)
        v1_grad = np.swapaxes(self.v.value, -1, -2) @ seed
        return v_grad, v1_grad


class Divide(BinaryOperator):
    def _backward(self, seed):
        return seed / self.v1.value, seed * -self.v.value / self.v1.value ** 2


class Transpose(UnaryOperator):
    def __init__(self, v, *dims):
        super().__init__(v)
        self.dims = dims

    def _backward(self, seed):
        if self.dims:
            return np.swapaxes(seed, *self.dims)
        else:
            return np.transpose(seed)


class Index(UnaryOperator):
    def __init__(self, v, index):
        super().__init__(v)
        self.index = index

    def _backward(self, seed):
        new_seed = np.zeros_like(self.v.value)
        np.add.at(new_seed, self.index, seed)
        return new_seed


class Concatenate(Operator):
    def __init__(self, dim, values):
        super().__init__(*values)
        self.dim = dim

    def backward(self, seed):
        sections = np.cumsum([v.shape[self.dim] for v in self.inputs])[:-1]
        split_seed = np.split(seed, axis=self.dim, indices_or_sections=sections)
        for s, v in zip(split_seed, self.inputs):
            v.accumulate(s)


class Stack(Operator):
    def __init__(self, dim, inputs):
        super().__init__(*inputs)
        self.dim = dim

    def backward(self, seed):
        unstacked_seed = np.unstack(seed, axis=self.dim)
        for v, s in zip(self.inputs, unstacked_seed):
            v.accumulate(s)


class Reshape(UnaryOperator):
    def _backward(self, seed):
        return np.reshape(seed, self.v.value.shape)


class Sum(UnaryOperator):
    def __init__(self, v, dim, keepdims):
        super().__init__(v)
        self.dim = dim
        self.keepdims = keepdims

    def _backward(self, seed):
        if self.dim is None or self.keepdims:
            return seed * np.ones_like(self.v.value)
        else:
            return np.repeat(np.expand_dims(seed, self.dim), self.v.shape[self.dim], self.dim)


class Sigmoid(UnaryOperator):
    def _backward(self, seed):
        y = 1 / (1 + np.exp(-self.v.value))
        y *= (1 - y)
        return seed * y


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


class Conv2d(BinaryOperator):

    def get_seed_padding(self, seed):
        k_size = np.array(np.flip(self.v1).shape[-2:])
        s_size = np.array(seed.shape)[-2:]
        v_shape = np.array(self.v.shape)[-2:]
        pad_x, pad_y = (k_size + v_shape - s_size) / 2
        return (int(pad_x), int(pad_x)), (int(pad_y), int(pad_y))

    def _backward(self, seed):
        grad_k = []
        # For each output channel, convolve with all input channels
        for s in seed:
            for x in self.v.value:
                grad_k.append(_conv2d_f(np.expand_dims(x, [0, 1]),
                                        np.expand_dims(s, [0, 1])))
        v1_grad = np.array(grad_k).squeeze().reshape(self.v1.shape)

        grad_c = [[] for _ in range(self.v.shape[0])]
        # For each pair of output and kernel, get the derivatives of each channel
        for s, k in zip(seed, self.v1.value):
            for i, k_c in enumerate(k):
                k_c = np.flip(k_c)
                padded_seed = np.pad(s, self.get_seed_padding(s))
                grad_c[i].append(_conv2d_f(np.expand_dims(padded_seed, [0, 1]),
                                           np.expand_dims(k_c, [0, 1])))
        # Sum the derivatives channel wise
        grad_x = np.sum(np.array(grad_c), 1)
        grad_x = np.squeeze(grad_x)
        v_grad = np.array(grad_x, ndmin=len(self.v.shape))
        return v_grad, v1_grad


class Log(UnaryOperator):
    def _backward(self, seed):
        return seed / (self.v.value + sys.float_info.epsilon)


class ReLU(UnaryOperator):
    def _backward(self, seed):
        grad = np.ones_like(self.v.value)
        grad[self.v.value <= 0] = 0
        return seed * grad


class Exp(UnaryOperator):
    def _backward(self, seed):
        return seed * np.exp(self.v.value)


class Sqrt(UnaryOperator):
    def _backward(self, seed):
        return seed / (2 * np.sqrt(self.v.value))


class Mean(UnaryOperator):
    def __init__(self, v, dim):
        super().__init__(v)
        self.dim = dim

    def _backward(self, seed):
        n = self.v.shape[self.dim] if self.dim is not None else self.v.value.size
        grad = np.ones_like(self.v.value) / n
        return seed * grad


class Mask(UnaryOperator):
    def __init__(self, v, mask):
        super().__init__(v)
        self.mask = mask

    def _backward(self, seed):
        return seed * self.mask
