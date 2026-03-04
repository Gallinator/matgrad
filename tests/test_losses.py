import unittest

import numpy as np
import torch
from torch import tensor, Tensor

from autodiff.variable import Variable
from nn.losses import mse_loss, bce_loss, cross_entropy


def assert_tensor_close(a: Variable, b: Tensor, atol=0.0):
    np.testing.assert_allclose(a.value.squeeze(), b.numpy(force=True), atol=atol)


class TestGradBase(unittest.TestCase):
    def setUp(self):
        self.input = Variable(np.array([[0.1, 0.9],
                                        [0.5, 1.0]]),
                              requires_grad=True)
        self.target = Variable(np.array([[1.0, 0.0],
                                         [0.0, 1.0]]),
                               requires_grad=True)

        self.input_t = tensor(self.input.value, requires_grad=True)
        self.target_t = tensor(self.target.value, requires_grad=True)

    def test_mse(self):
        actual = mse_loss(self.input, self.target)
        expected = torch.nn.functional.mse_loss(self.input_t, self.target_t)

        assert_tensor_close(actual, expected)

    def test_bce(self):
        actual = bce_loss(self.input, self.target)
        expected = torch.nn.functional.binary_cross_entropy(self.input_t, self.target_t)

        assert_tensor_close(actual, expected, 1e-15)

    def test_cross_entropy(self):
        c = Variable(np.arange(2))
        c_t = torch.tensor(c.value)

        actual = cross_entropy(self.input, c)
        expected = torch.nn.functional.cross_entropy(self.input_t, c_t)

        assert_tensor_close(actual, expected, 1e-15)
