import numpy as np
import pytest
import torch
from torch import tensor, Tensor

import autodiff.functions as af

from autodiff import variable
from autodiff.variable import Variable, transpose

A_PARAMS = [variable.random(2, 3, 2, 4),
            variable.random(2, 4),
            variable.random(1, 4),
            variable.random(2, 1),
            variable.random(1, 1),
            variable.random(4),
            variable.random(1)]

B_PARAMS = [variable.random(2, 3, 2, 4),
            variable.random(2, 4),
            variable.random(1, 4),
            variable.random(2, 1),
            variable.random(1, 1),
            variable.random(4),
            variable.random(1)]


def assert_close(v: Variable, t: Tensor, atol=1e-15):
    np.testing.assert_allclose(v.value, t.numpy(force=True), atol=atol)


def assert_equal(v: Variable, t: Tensor):
    np.testing.assert_array_equal(v.value, t.numpy(force=True))


@pytest.mark.parametrize('a', [Variable(np.ones((2, 3, 2, 4)) * -1e-15)])
@pytest.mark.parametrize('dim', [0, 1, 2, 3])
def test_softmax(a, dim):
    a_torch = tensor(a.value, requires_grad=True)

    assert_close(af.softmax(a, dim), torch.softmax(a_torch, dim), 1e-15)


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 4),
                               variable.random(2, 4),
                               variable.random(1, 4)])
@pytest.mark.parametrize('b', [variable.random(2, 3, 4, 2),
                               variable.random(4, 2),
                               variable.random(4, 1)])
def test_matmul(a, b):
    a_torch, b_torch = tensor(a.value, requires_grad=True), tensor(b.value, requires_grad=True)

    assert_close(a @ b, a_torch @ b_torch)


@pytest.mark.parametrize('a', [Variable(np.ones((2, 3, 2, 4)))])
@pytest.mark.parametrize('mask', [np.random.choice([True, False], size=(2, 3, 2, 4))])
def test_masked_fill(a, mask):
    a_torch = tensor(a.value)
    mask_torch = tensor(mask)

    assert_close(af.masked_fill(a, mask, 1e-15), torch.masked_fill(a_torch, mask_torch, 1e-15), 1e-15)


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 4)])
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize('p', [0.0, 1.0])
def test_dropout(a, train, p):
    a_torch = tensor(a.value, requires_grad=True)
    actual = af.dropout(a, p, train)
    expected = torch.dropout(a_torch, p, train)

    assert_close(actual, expected)


@pytest.mark.parametrize('a', [Variable(np.ones((2, 3, 2, 4)))])
@pytest.mark.parametrize('dim', [(0, 1), (2, 3), (-1, -2), (1, 3)])
def test_transpose(a, dim):
    a_torch = tensor(a.value, requires_grad=True)

    assert_close(transpose(a, *dim), torch.transpose(a_torch, *dim), 1e-15)


@pytest.mark.parametrize('a', [Variable(np.ones((3, 2, 4)))])
@pytest.mark.parametrize("shape", [(3, 2, 2, 2)])
def test_reshape(a, shape):
    a_torch = tensor(a.value, requires_grad=True)

    assert_close(af.reshape(a, shape), torch.reshape(a_torch, shape), 1e-15)
