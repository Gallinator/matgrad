import numpy as np
import pytest
import torch
from torch import tensor, Tensor
import autodiff.functions as af
from autodiff import variable
from autodiff.functions import dropout
from autodiff.variable import Variable, transpose


def assert_grad_equal(tensor: Tensor, v: Variable):
    np.testing.assert_array_equal(v.grad, tensor.grad.numpy(force=True))


def assert_grad_close(tensor: Tensor, v: Variable, atol=1e-15):
    np.testing.assert_allclose(v.grad, tensor.grad.numpy(force=True), atol=atol)


def zero_grads(*args):
    for a in args:
        a.grad = None


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


@pytest.mark.parametrize('a', A_PARAMS)
@pytest.mark.parametrize('b', B_PARAMS)
def test_add(a, b):
    a_torch, b_torch = tensor(a.value, requires_grad=True), tensor(b.value, requires_grad=True)

    (a + b).backward()
    (a_torch + b_torch).sum().backward()

    assert_grad_close(a_torch, a)
    assert_grad_close(b_torch, b)

    zero_grads(a, b)


@pytest.mark.parametrize('a', A_PARAMS)
@pytest.mark.parametrize('b', B_PARAMS)
def test_mul(a, b):
    a_torch, b_torch = tensor(a.value, requires_grad=True), tensor(b.value, requires_grad=True)
    (a * b).backward()
    (a_torch * b_torch).sum().backward()

    assert_grad_close(a_torch, a)
    assert_grad_close(b_torch, b)

    zero_grads(a, b)


@pytest.mark.parametrize('a', A_PARAMS)
@pytest.mark.parametrize('b', B_PARAMS)
def test_sub(a, b):
    a_torch, b_torch = tensor(a.value, requires_grad=True), tensor(b.value, requires_grad=True)
    (a - b).backward()
    (a_torch - b_torch).sum().backward()

    assert_grad_close(a_torch, a)
    assert_grad_close(b_torch, b)

    zero_grads(a, b)


@pytest.mark.parametrize('a', A_PARAMS)
@pytest.mark.parametrize('b', B_PARAMS)
def test_truediv(a, b):
    a_torch, b_torch = tensor(a.value, requires_grad=True), tensor(b.value, requires_grad=True)
    (a / b).backward()
    (a_torch / b_torch).sum().backward()

    assert_grad_close(a_torch, a)
    assert_grad_close(b_torch, b)

    zero_grads(a, b)


@pytest.mark.parametrize('a', A_PARAMS)
@pytest.mark.parametrize('e', [-5, -1, 0, 1, 5])
def test_pow(a, e):
    a_torch = tensor(a.value, requires_grad=True)
    (a ** e).backward()
    (a_torch ** e).sum().backward()

    assert_grad_close(a_torch, a)

    zero_grads(a)


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 4),
                               variable.random(2, 4),
                               variable.random(1, 4)])
@pytest.mark.parametrize('b', [variable.random(2, 3, 4, 2),
                               variable.random(4, 2),
                               variable.random(4, 1)])
def test_matmul(a, b):
    a_torch, b_torch = tensor(a.value, requires_grad=True), tensor(b.value, requires_grad=True)
    (a @ b).backward()
    (a_torch @ b_torch).sum().backward()

    assert_grad_close(a_torch, a)
    assert_grad_close(b_torch, b)

    zero_grads(a, b)


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 1),
                               variable.random(2, 1),
                               variable.random(1, 1)])
@pytest.mark.parametrize('b', [variable.random(2, 3, 1, 2),
                               variable.random(1, 2),
                               variable.random(1, 1)])
def test_matmul_col_vec(a, b):
    a_torch, b_torch = tensor(a.value, requires_grad=True), tensor(b.value, requires_grad=True)
    (a @ b).backward()
    (a_torch @ b_torch).sum().backward()

    assert_grad_close(a_torch, a)
    assert_grad_close(b_torch, b)

    zero_grads(a, b)


@pytest.mark.parametrize('a', A_PARAMS)
def test_neg(a):
    a_torch = tensor(a.value, requires_grad=True)
    (-a).backward()
    (-a_torch).sum().backward()

    assert_grad_close(a_torch, a)

    zero_grads(a)


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 4)])
@pytest.mark.parametrize('dims', [tuple(), (-1, -2), (0, 2)])
def test_transpose(a, dims):
    a_torch = tensor(a.value, requires_grad=True)
    transpose(a, *dims).backward()
    if dims:
        a_torch.transpose(*dims).sum().backward()
    else:
        a_torch.T.sum().backward()

    assert_grad_close(a_torch, a)

    zero_grads(a)


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 4) * Variable(np.array([-1000]))])
@pytest.mark.parametrize('dim', [0, 1, 2, 3])
def test_softmax(a, dim):
    af.softmax(a, dim).backward()
    a_torch = tensor(a.value, requires_grad=True)
    torch.softmax(a_torch, dim).sum().backward()

    assert_grad_close(a_torch, a, 1e-15)


@pytest.mark.parametrize('a', [variable.random(5, 3, 2, 4)])
def test_indexing_single(a):
    a[0].backward()
    a_torch = tensor(a.value, requires_grad=True)
    a_torch[0].sum().backward()

    assert_grad_equal(a_torch, a)


@pytest.mark.parametrize('a', [variable.random(5, 3, 2, 4)])
def test_indexing_slice(a):
    a[0:2, ...].backward()
    a_torch = tensor(a.value, requires_grad=True)
    a_torch[0:2, ...].sum().backward()

    assert_grad_equal(a_torch, a)


@pytest.mark.parametrize('a', [variable.random(5, 3, 2, 4)])
def test_indexing_np(a):
    idx = np.array([[1, 2], [2, 1]])
    a[idx].backward()
    a_torch = tensor(a.value, requires_grad=True)
    a_torch[tensor(idx)].sum().backward()

    assert_grad_equal(a_torch, a)


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 4)])
@pytest.mark.parametrize('shape', [(2, 3, 4, 2), (6, 4, 2)])
def test_reshape(a, shape):
    af.reshape(a, shape).backward()
    a_torch = tensor(a.value, requires_grad=True)
    torch.reshape(a_torch, shape).sum().backward()

    assert_grad_close(a_torch, a, 1e-15)

    zero_grads(a)


@pytest.mark.parametrize('a', [Variable(np.ones((2, 3, 2, 4)), requires_grad=True)])
@pytest.mark.parametrize('mask', [np.random.choice([True, False], size=(2, 3, 2, 4))])
def test_masked_fill(a, mask):
    a_torch = tensor(a.value, requires_grad=True)
    mask_torch = tensor(mask, dtype=torch.bool)
    af.masked_fill(a, mask, 1e-15).backward()
    torch.masked_fill(a_torch, mask_torch, 1e-15).sum().backward()

    assert_grad_close(a_torch, a, 1e-15)


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 4)])
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize('p', [0.0, 1.0])
def test_dropout(a, train, p):
    a_torch = tensor(a.value, requires_grad=True)
    dropout(a, p, train).backward()
    torch.dropout(a_torch, p, train).sum().backward()

    assert_grad_close(a_torch, a)

    zero_grads(a)


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 4),
                               variable.random(3, 2, 4)])
def test_layer_norm(a):
    norm_shape = a.shape[-1:]
    w = Variable(np.random.rand(*norm_shape), requires_grad=True)
    b = Variable(np.random.rand(*norm_shape), requires_grad=True)

    w_t = tensor(w.value, requires_grad=True)
    b_t = tensor(b.value, requires_grad=True)

    a_t = tensor(a.value, requires_grad=True)
    af.layer_norm(a, w, b).backward()
    torch.layer_norm(a_t, norm_shape, w_t, b_t).sum().backward()

    assert_grad_close(a_t, a, 1e-15)
    assert_grad_close(w_t, w, 1e-15)
    assert_grad_close(b_t, b, 1e-15)

    zero_grads(a)


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 4)])
@pytest.mark.parametrize('dim', [0, 1, 2, 3])
def test_mean(a, dim):
    af.mean(a, dim).backward()
    a_torch = tensor(a.value, requires_grad=True)
    torch.mean(a_torch, dim, keepdim=True).sum().backward()

    assert_grad_close(a_torch, a, 1e-15)

    zero_grads(a)


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 4)])
@pytest.mark.parametrize('b', [variable.random(2, 3, 2, 4)])
@pytest.mark.parametrize('dim', [0, 1, 2, -1])
def test_cat(a, b, dim):
    af.cat([a, b], dim).backward()
    a_torch = tensor(a.value, requires_grad=True)
    b_torch = tensor(b.value, requires_grad=True)
    torch.cat([a_torch, b_torch], dim).sum().backward()

    assert_grad_close(a_torch, a, 1e-15)
    assert_grad_close(b_torch, b, 1e-15)

    zero_grads(a, b)


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 4)])
@pytest.mark.parametrize('b', [variable.random(2, 3, 2, 4)])
@pytest.mark.parametrize('dim', [0, 1, 2, -1])
def test_stack(a, b, dim):
    af.stack([a, b], dim).backward()
    a_torch = tensor(a.value, requires_grad=True)
    b_torch = tensor(b.value, requires_grad=True)
    torch.stack([a_torch, b_torch], dim).sum().backward()

    assert_grad_close(a_torch, a, 1e-15)
    assert_grad_close(b_torch, b, 1e-15)

    zero_grads(a, b)


@pytest.mark.parametrize('a', A_PARAMS)
def test_sin(a):
    a_torch = tensor(a.value, requires_grad=True)
    af.sin(a).backward()
    torch.sin(a_torch).sum().backward()

    assert_grad_close(a_torch, a)

    zero_grads(a)


@pytest.mark.parametrize('a', A_PARAMS)
def test_cos(a):
    a_torch = tensor(a.value, requires_grad=True)
    af.cos(a).backward()
    torch.cos(a_torch).sum().backward()

    assert_grad_close(a_torch, a)

    zero_grads(a)


@pytest.mark.parametrize('a', A_PARAMS)
def test_sigmoid(a):
    a_torch = tensor(a.value, requires_grad=True)
    af.sigmoid(a).backward()
    torch.sigmoid(a_torch).sum().backward()

    assert_grad_close(a_torch, a)

    zero_grads(a)


@pytest.mark.parametrize('a', A_PARAMS)
def test_log(a):
    a_torch = tensor(a.value, requires_grad=True)
    af.log(a).backward()
    torch.log(a_torch).sum().backward()

    assert_grad_close(a_torch, a)

    zero_grads(a)


@pytest.mark.parametrize('a', A_PARAMS)
def test_relu(a):
    a_torch = tensor(a.value, requires_grad=True)
    af.relu(a).backward()
    torch.relu(a_torch).sum().backward()

    assert_grad_close(a_torch, a)

    zero_grads(a)


@pytest.mark.parametrize('a', A_PARAMS)
def test_exp(a):
    a_torch = tensor(a.value, requires_grad=True)
    af.exp(a).backward()
    torch.exp(a_torch).sum().backward()

    assert_grad_close(a_torch, a)

    zero_grads(a)


@pytest.mark.parametrize('a', A_PARAMS)
def test_sqrt(a):
    a_torch = tensor(a.value, requires_grad=True)
    af.sqrt(a).backward()
    torch.sqrt(a_torch).sum().backward()

    assert_grad_close(a_torch, a)

    zero_grads(a)


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 4)])
@pytest.mark.parametrize('dim', [0, 1, 2, 3])
@pytest.mark.parametrize('keepdims', [True, False])
def test_sum(a, dim, keepdims):
    af.sum(a, dim, keepdims=keepdims).backward()
    a_torch = tensor(a.value, requires_grad=True)
    torch.sum(a_torch, dim, keepdim=keepdims).sum().backward()

    assert_grad_close(a_torch, a, 1e-15)

    zero_grads(a)


@pytest.mark.parametrize('img,k', [(variable.random(3, 4, 4), variable.random(1, 3, 2, 2)),
                                   (variable.random(3, 4, 4), variable.random(4, 3, 2, 2))])
def test_conv2d(img, k):
    img_t = tensor(img.value, requires_grad=True)
    k_t = tensor(k.value, requires_grad=True)

    af.conv2d(img, k).backward()
    torch.conv2d(img_t, k_t).sum().backward()

    assert_grad_close(img_t, img, 1e-15)
    assert_grad_close(k_t, k, 1e-15)

    zero_grads(img, k)
