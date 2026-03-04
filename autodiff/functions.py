import sys

import numpy as np
from autodiff.grad_functions import Reshape, Concatenate, Stack, Sin, Sigmoid, Log, Sum, ReLU, Conv2d, _conv2d_f, Exp, \
    Mean, Cos, Sqrt
from autodiff.variable import Variable, any_requires_grad


def reshape(v: Variable, shape):
    return Variable(np.reshape(v.value, shape), Reshape(v), v.requires_grad)


def cat(values, dim):
    return Variable(np.concatenate([v.value for v in values], axis=dim), Concatenate(dim, *values),
                    any_requires_grad(*values))


def stack(values, dim):
    return Variable(np.stack([v.value for v in values], axis=dim), Stack(dim, *values),
                    any_requires_grad(*values))


def sin(v: Variable):
    return Variable(np.sin(v.value), Sin(v), v.requires_grad)

def cos(v: Variable):
    return Variable(np.cos(v.value), Cos(v), v.requires_grad)

def sigmoid(v: Variable):
    return Variable(1 / (1 + np.exp(-v.value)), Sigmoid(v), v.requires_grad)


def log(v: Variable):
    return Variable(np.log(v.value + sys.float_info.epsilon), Log(v), v.requires_grad)


def sum(v: Variable, dim=None, keepdims=False):
    return Variable(np.sum(v.value, axis=dim, keepdims=keepdims), Sum(v, dim, keepdims), v.requires_grad)


def relu(v: Variable):
    return Variable(np.array(np.maximum(0.0, v.value)), ReLU(v), v.requires_grad)


def exp(v: Variable):
    return Variable(np.exp(v.value), Exp(v), v.requires_grad)


def sqrt(v: Variable):
    return Variable(np.sqrt(v.value), Sqrt(v), v.requires_grad)


def softmax(v: Variable, dim):
    e = exp(v)
    return e / sum(e, dim, keepdims=True)


def mean(v: Variable):
    return Variable(np.mean(v, keepdims=True), Mean(v), v.requires_grad)


def conv2d(v: Variable, k: Variable):
    return Variable(_conv2d_f(v.value, k.value), Conv2d(v, k), any_requires_grad(v, k))
