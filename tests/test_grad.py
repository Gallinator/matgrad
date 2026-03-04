import unittest

import numpy as np
import torch
from torch import tensor, Tensor

import autodiff.functions as af
from autodiff.grad_functions import broadcast_grad
from autodiff.variable import Variable


def assert_grad_equal(tensor: Tensor, v: Variable):
    np.testing.assert_array_equal(v.grad, tensor.grad.numpy(force=True))


def assert_grad_close(tensor: Tensor, v: Variable, atol):
    np.testing.assert_allclose(v.grad, tensor.grad.numpy(force=True), atol=atol)


class TestGradBase(unittest.TestCase):
    def setUp(self):
        self.a = Variable(np.array([[1.0, 2.0],
                                    [-1.0, 1.0]]),
                          requires_grad=True)
        self.b = Variable(np.array([[0.0, 1.0],
                                    [-2.0, 3.0]]),
                          requires_grad=True)
        self.v = Variable(np.array([[0.0, 1.0]]), requires_grad=True)
        self.s = Variable(np.array([[2.0]]), requires_grad=True)

        self.a_t = tensor(self.a.value, requires_grad=True)
        self.b_t = tensor(self.b.value, requires_grad=True)
        self.v_t = tensor(self.v.value, requires_grad=True)
        self.s_t = tensor(self.s.value, requires_grad=True)


class TestGradOperators(TestGradBase):

    def test_broadcast_grad_vec(self):
        grad = self.a.value
        v = self.v.value
        expected = np.array([[0, 3]])

        np.testing.assert_array_equal(broadcast_grad(grad, v), expected)

    def test_broadcast_grad_vec_t(self):
        grad = self.a.value
        v = self.v.value.T
        expected = np.array([[3], [0]])

        np.testing.assert_array_equal(broadcast_grad(grad, v), expected)

    def test_broadcast_grad_mat(self):
        grad = np.ones((2, 2))
        v = np.ones((2, 2))
        expected = np.array([[1, 1],
                             [1, 1]])

        np.testing.assert_array_equal(broadcast_grad(grad, v), expected)

    def test_broadcast_grad_scalar(self):
        grad = np.ones((2, 2))
        v = np.ones((1, 1))
        expected = np.array([[4]])

        np.testing.assert_array_equal(broadcast_grad(grad, v), expected)

    def test_sum_grad(self):
        (self.a + self.b).backward()
        (self.a_t + self.b_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.b_t, self.b)

    def test_sum_grad_vec(self):
        (self.a + self.v).backward()
        (self.a_t + self.v_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.v_t, self.v)

    def test_sum_grad_scalar(self):
        (self.a + self.s).backward()
        (self.a_t + self.s_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.s_t, self.s)

    def test_sub_grad(self):
        (self.a - self.b).backward()
        (self.a_t - self.b_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.b_t, self.b)

    def test_sub_grad_vec(self):
        (self.a - self.v).backward()
        (self.a_t - self.v_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.v_t, self.v)

    def test_sub_grad_scalar(self):
        (self.a - self.s).backward()
        (self.a_t - self.s_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.s_t, self.s)

    def test_truediv_grad(self):
        (self.a / self.b).backward()
        (self.a_t / self.b_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.b_t, self.b)

    def test_truediv_vec(self):
        (self.a / self.v).backward()
        (self.a_t / self.v_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.v_t, self.v)

    def test_truediv_scalar(self):
        (self.a / self.s).backward()
        (self.a_t / self.s_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.s_t, self.s)

    def test_pow_pos(self):
        (self.a ** 2).backward()
        (self.a_t ** 2).sum().backward()

        assert_grad_equal(self.a_t, self.a)

    def test_pow_zero(self):
        (self.a ** 0).backward()
        (self.a_t ** 0).sum().backward()

        assert_grad_equal(self.a_t, self.a)

    def test_pow_neg(self):
        (self.a ** -1).backward()
        (self.a_t ** -1).sum().backward()

        assert_grad_equal(self.a_t, self.a)

    def test_mul(self):
        (self.a * self.b).backward()
        (self.a_t * self.b_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.b_t, self.b)

    def test_mul_vec(self):
        (self.a * self.v).backward()
        (self.a_t * self.v_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.v_t, self.v)

    def test_mul_scalar(self):
        (self.a * self.s).backward()
        (self.a_t * self.s_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.s_t, self.s)

    def test_matmul_mat_mat(self):
        (self.a @ self.b).backward()
        (self.a_t @ self.b_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.b_t, self.b)

    def test_matmul_vec_mat(self):
        (self.v @ self.b).backward()
        (self.v_t @ self.b_t).sum().backward()

        assert_grad_equal(self.b_t, self.b)
        assert_grad_equal(self.v_t, self.v)

    def test_matmul_vec_vec(self):
        (self.v @ self.v.T).backward()
        (self.v_t @ self.v_t.T).sum().backward()

        assert_grad_equal(self.v_t, self.v)

    def test_matmul_vec_scal(self):
        (self.v.T @ self.s).backward()
        (self.v_t.T @ self.s_t).sum().backward()

        assert_grad_equal(self.v_t, self.v)
        assert_grad_equal(self.s_t, self.s)

    def test_neg_grad(self):
        (-self.a).backward()
        (-self.a_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)

    def test_transpose_grad(self):
        self.a.T.backward()
        self.a_t.T.sum().backward()

        assert_grad_equal(self.a_t, self.a)

    def test_indexing_single_grad(self):
        self.a[0].backward()
        self.a_t[0].sum().backward()

        assert_grad_equal(self.a_t, self.a)

    def test_indexing_slice_grad(self):
        self.a[0:1, 1:].backward()
        self.a_t[0:1, 1:].sum().backward()

        assert_grad_equal(self.a_t, self.a)


class TestGradFunctions(TestGradBase):
    def test_reshape_grad(self):
        af.reshape(self.a, (1, 4)).backward()
        torch.reshape(self.a_t, (1, 4)).sum().backward()

        assert_grad_equal(self.a_t, self.a)

    def test_cat_grad_dim0(self):
        af.cat([self.a, self.b], dim=0).backward()
        torch.cat([self.a_t, self.b_t], dim=0).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.b_t, self.b)

    def test_cat_grad_dim1(self):
        af.cat([self.a, self.b], dim=1).backward()
        torch.cat([self.a_t, self.b_t], dim=1).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.b_t, self.b)

    def test_cat_stack_dim0(self):
        af.stack([self.a, self.b], dim=0).backward()
        torch.stack([self.a_t, self.b_t], dim=0).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.b_t, self.b)

    def test_cat_stack_dim1(self):
        af.stack([self.a, self.b], dim=1).backward()
        torch.stack([self.a_t, self.b_t], dim=1).sum().backward()

        assert_grad_equal(self.a_t, self.a)
        assert_grad_equal(self.b_t, self.b)

    def test_sin(self):
        af.sin(self.a).backward()
        torch.sin(self.a_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)

    def test_sigmoid(self):
        af.sigmoid(self.a).backward()
        torch.sigmoid(self.a_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)

    def test_log(self):
        af.log(self.a).backward()
        torch.log(self.a_t).sum().backward()

        # Numpy gives slightly different gradients due to NaNs handling
        assert_grad_close(self.a_t, self.a, 1e-15)

    def test_sum(self):
        af.sum(self.a).backward()
        torch.sum(self.a_t).backward()

        assert_grad_equal(self.a_t, self.a)

    def test_sum_dim1(self):
        af.sum(self.a, dim=1).backward()
        torch.sum(self.a_t, dim=1).sum().backward()

        assert_grad_equal(self.a_t, self.a)

    def test_sum_keep(self):
        af.sum(self.a, dim=1, keepdims=True).backward()
        torch.sum(self.a_t, dim=1, keepdim=True).sum().backward()

        assert_grad_equal(self.a_t, self.a)

    def test_relu(self):
        af.relu(self.a).backward()
        torch.relu(self.a_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)

    def test_exp(self):
        af.exp(self.a).backward()
        torch.exp(self.a_t).sum().backward()

        assert_grad_equal(self.a_t, self.a)

    def test_softmax_dim0(self):
        af.softmax(self.a, 0).backward()
        torch.softmax(self.a_t, 0).sum().backward()

        assert_grad_close(self.a_t, self.a, 1e-15)

    def test_softmax_dim1(self):
        af.softmax(self.a, 1).backward()
        torch.softmax(self.a_t, 1).sum().backward()

        assert_grad_close(self.a_t, self.a, 1e-15)

    def test_conv2d(self):
        img = Variable(np.random.rand(3, 4, 4), requires_grad=True)
        img_t = tensor(img.value, requires_grad=True)

        k = Variable(np.random.rand(1, 3, 2, 2), requires_grad=True)
        k_t = tensor(k.value, requires_grad=True)

        af.conv2d(img, k).backward()
        torch.conv2d(img_t, k_t).sum().backward()

        assert_grad_close(img_t, img, 1e-15)
        assert_grad_close(k_t, k, 1e-15)

    def test_conv2d_multi_kernel(self):
        img = Variable(np.random.rand(3, 4, 4), requires_grad=True)
        img_t = tensor(img.value, requires_grad=True)

        k = Variable(np.random.rand(4, 3, 2, 2), requires_grad=True)
        k_t = tensor(k.value, requires_grad=True)

        af.conv2d(img, k).backward()
        torch.conv2d(img_t, k_t).sum().backward()

        assert_grad_close(img_t, img, 1e-15)
        assert_grad_close(k_t, k, 1e-15)
