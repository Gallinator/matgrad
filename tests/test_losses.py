import numpy as np
import pytest
import torch
from torch import tensor, Tensor

import nn.losses as al
from autodiff import variable
from autodiff.variable import Variable

INPUT = [variable.random(2, 3, 2, 4),
         variable.random(2, 4),
         variable.random(1, 4),
         variable.random(2, 1),
         variable.random(1, 1),
         variable.random(4),
         variable.random(1)]

TARGET = [variable.random(2, 3, 2, 4),
          variable.random(2, 4),
          variable.random(1, 4),
          variable.random(2, 1),
          variable.random(1, 1),
          variable.random(4),
          variable.random(1)]


def assert_tensor_close(a: Variable, b: Tensor, atol=0.0):
    np.testing.assert_allclose(a.value.squeeze(), b.numpy(force=True), atol=atol)


def assert_grad_equal(tensor: Tensor, v: Variable):
    np.testing.assert_array_equal(v.grad, tensor.grad.numpy(force=True))


def assert_grad_close(tensor: Tensor, v: Variable, atol=1e-15):
    np.testing.assert_allclose(v.grad, tensor.grad.numpy(force=True), atol=atol)


def zero_grads(*args):
    for a in args:
        a.grad = None


@pytest.mark.parametrize('a', [Variable(np.random.rand(3, 4, 10))])
def test_cross_entropy(a):
    a_tensor = tensor(a.value)
    c = Variable(np.random.randint(0, 4, size=(3, 10)))
    c_t = tensor(c.value)
    actual = al.cross_entropy(a, c)
    expected = torch.nn.functional.cross_entropy(a_tensor, c_t)

    assert_tensor_close(actual, expected, 1e-15)


@pytest.mark.parametrize('a', [Variable(np.random.rand(3, 4, 10), requires_grad=True)])
def test_cross_entropy_grad(a):
    a_tensor = tensor(a.value, requires_grad=True)
    c = Variable(np.random.randint(0, 4, size=(3, 10)))
    c_tensor = tensor(c.value)
    al.cross_entropy(a, c).backward()
    torch.nn.functional.cross_entropy(a_tensor, c_tensor).backward()

    assert_grad_close(a_tensor, a)


@pytest.mark.parametrize('input,target', zip(INPUT, TARGET))
def test_mse(input, target):
    in_tensor = tensor(input.value)
    tgt_t = tensor(target.value)
    actual = al.mse_loss(input, target)
    expected = torch.nn.functional.mse_loss(in_tensor, tgt_t)

    assert_tensor_close(actual, expected, 1e-15)


@pytest.mark.parametrize('input,target', zip(INPUT, TARGET))
def test_mse(input, target):
    in_tensor = tensor(input.value, requires_grad=True)
    tgt_tensor = tensor(target.value, requires_grad=True)
    al.mse_loss(input, target).backward()
    torch.nn.functional.mse_loss(in_tensor, tgt_tensor).backward()

    assert_grad_close(in_tensor, input)
    assert_grad_close(tgt_tensor, target)

    zero_grads(input, target)


@pytest.mark.parametrize('input,target', zip(INPUT, TARGET))
def test_bce(input, target):
    in_tensor = tensor(input.value)
    tgt_t = tensor(target.value)
    actual = al.bce_loss(input, target)
    expected = torch.nn.functional.binary_cross_entropy(in_tensor, tgt_t)

    assert_tensor_close(actual, expected, 1e-15)


@pytest.mark.parametrize('input,target', zip(INPUT, TARGET))
def test_bce_grad(input, target):
    in_tensor = tensor(input.value, requires_grad=True)
    tgt_tensor = tensor(target.value, requires_grad=True)
    al.bce_loss(input, target).backward()
    torch.nn.functional.binary_cross_entropy(in_tensor, tgt_tensor).backward()

    assert_grad_close(in_tensor, input)
    assert_grad_close(tgt_tensor, target)

    zero_grads(input, target)
