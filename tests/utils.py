import numpy as np
from torch import Tensor

from autodiff.variable import Variable


def assert_grad_equal(actual: Variable, desired: Tensor):
    np.testing.assert_array_equal(actual.grad, desired.grad.numpy(force=True))


def assert_grad_close(actual: Variable, desired: Tensor, atol=1e-15):
    np.testing.assert_allclose(actual.grad, desired.grad.numpy(force=True), atol=atol)


def assert_close(actual: Variable, desired: Tensor, atol=1e-15):
    np.testing.assert_allclose(actual.value, desired.numpy(force=True), atol=atol)


def assert_equal(actual: Variable, desired: Tensor):
    np.testing.assert_array_equal(actual.value, desired.numpy(force=True))


def zero_grads(*args):
    for a in args:
        a.grad = None
