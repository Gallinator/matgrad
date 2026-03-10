import numpy as np
import pytest
import torch
from torch import tensor
from torch.nn.parameter import Parameter

from autodiff import variable
from nn.modules import attention, Linear, LayerNorm, Embedding, MultiHeadAttention, xavier_init
from tests.torch_multi_head_attention import TorchMultiHeadAttention
from tests.utils import assert_close, assert_grad_close, zero_grads, assert_equal, assert_grad_equal


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 4), variable.random(3, 5)])
def test_attention(a):
    actual = attention(a, a, a)
    a_tensor = tensor(a.value)

    expected = torch.nn.functional.scaled_dot_product_attention(a_tensor, a_tensor, a_tensor)

    assert_close(actual, expected)


@pytest.mark.parametrize('a', [variable.random(2, 3, 2, 4), variable.random(3, 5)])
def test_attention_grad(a):
    att = attention(a, a, a)
    att.backward()
    a_tensor = tensor(a.value, requires_grad=True)

    torch.nn.functional.scaled_dot_product_attention(a_tensor, a_tensor, a_tensor).sum().backward()

    assert_grad_close(a, a_tensor)

    zero_grads(a)


@pytest.fixture
def setup_linear_layers(request):
    a = request.param
    in_features, out_features = a.shape[-1], 3
    linear = Linear(in_features, out_features)
    torch_layer = torch.nn.Linear(in_features, out_features)

    # Since weights are initialized randomly, set them explicitly
    torch_layer.weight = Parameter(tensor(linear.w.value).squeeze(), requires_grad=True)
    torch_layer.bias = Parameter(tensor(linear.b.value).squeeze(), requires_grad=True)

    return a, tensor(a.value, requires_grad=True), linear, torch_layer


@pytest.mark.parametrize('setup_linear_layers', [variable.random(3, 10, 4), variable.random(10, 4)], indirect=True)
def test_linear(setup_linear_layers):
    a, torch_a, linear, torch_layer = setup_linear_layers
    assert_close(linear(a), torch_layer(torch_a))


@pytest.mark.parametrize('setup_linear_layers', [variable.random(3, 10, 4), variable.random(10, 4)], indirect=True)
def test_linear_grad(setup_linear_layers):
    a, torch_a, linear, torch_layer = setup_linear_layers
    linear(a).backward()
    torch_layer(torch_a).sum().backward()

    assert_grad_close(a, torch_a)
    assert_grad_close(linear.w, torch_layer.weight)
    assert_grad_close(linear.b, torch_layer.bias)


@pytest.fixture
def setup_layer_norm_layers(request):
    a = request.param
    in_features = a.shape[-1]
    layer = LayerNorm((in_features,))
    torch_layer = torch.nn.LayerNorm(in_features)

    # Since weights are initialized randomly, set them explicitly
    torch_layer.weight = Parameter(tensor(layer.w.value).squeeze(), requires_grad=True)
    torch_layer.bias = Parameter(tensor(layer.b.value).squeeze(), requires_grad=True)

    return a, tensor(a.value, requires_grad=True), layer, torch_layer


@pytest.mark.parametrize('setup_layer_norm_layers', [variable.random(2, 3, 10, 4)], indirect=True)
def test_layer_norm(setup_layer_norm_layers):
    a, torch_a, linear, torch_layer = setup_layer_norm_layers
    assert_close(linear(a), torch_layer(torch_a), 1e-12)


@pytest.mark.parametrize('setup_layer_norm_layers', [variable.random(2, 3, 10, 4)], indirect=True)
def test_layer_norm_grad(setup_layer_norm_layers):
    a, torch_a, linear, torch_layer = setup_layer_norm_layers
    linear(a).backward()
    torch_layer(torch_a).sum().backward()

    assert_grad_close(a, torch_a, 1e-12)
    assert_grad_close(linear.w, torch_layer.weight, 1e-12)
    assert_grad_close(linear.b, torch_layer.bias, 1e-12)


@pytest.fixture
def setup_embedding_layers(request):
    a = request.param
    layer = Embedding(20, 32)
    torch_layer = torch.nn.Embedding(20, 32)

    # Since weights are initialized randomly, set them explicitly
    torch_layer.weight = Parameter(tensor(layer.weight.value).squeeze(), requires_grad=True)

    return a, tensor(a.value), layer, torch_layer


@pytest.mark.parametrize('setup_embedding_layers', [variable.randint(0, 20, (2, 5))], indirect=True)
def test_embedding(setup_embedding_layers):
    a, torch_a, layer, torch_layer = setup_embedding_layers
    assert_equal(layer(a), torch_layer(torch_a))


@pytest.mark.parametrize('setup_embedding_layers', [variable.randint(0, 20, (2, 5))], indirect=True)
def test_embedding_grad(setup_embedding_layers):
    a, torch_a, layer, torch_layer = setup_embedding_layers
    layer(a).backward()
    torch_layer(torch_a).sum().backward()

    assert_grad_equal(layer.weight, torch_layer.weight)


@pytest.fixture
def setup_multi_head_attention_layers(request):
    a = request.param
    layer = MultiHeadAttention(64, 16, 4)
    torch_layer = TorchMultiHeadAttention(64, 16, 4)

    # Since weights are initialized randomly, set them explicitly
    torch_layer.w_q = Parameter(tensor(layer.w_q.value, dtype=torch.double).squeeze(), requires_grad=True)
    torch_layer.w_k = Parameter(tensor(layer.w_k.value, dtype=torch.double).squeeze(), requires_grad=True)
    torch_layer.w_v = Parameter(tensor(layer.w_v.value, dtype=torch.double).squeeze(), requires_grad=True)
    torch_layer.w_o = Parameter(tensor(layer.w_o.value, dtype=torch.double).squeeze(), requires_grad=True)

    return a, tensor(a.value, dtype=torch.double), layer, torch_layer


@pytest.mark.parametrize('setup_multi_head_attention_layers', [variable.random(3, 10, 64)], indirect=True)
@pytest.mark.parametrize('mask', [None,
                                  np.random.choice([True, False], size=(3, 4, 10, 10))])
def test_multi_head_attention(setup_multi_head_attention_layers, mask):
    a, torch_a, layer, torch_layer = setup_multi_head_attention_layers
    torch_mask = tensor(mask, dtype=torch.bool) if mask is not None else mask

    assert_close(layer(a, a, a, mask), torch_layer(torch_a, torch_a, torch_a, torch_mask))


@pytest.mark.parametrize('setup_multi_head_attention_layers', [variable.random(3, 10, 64)], indirect=True)
@pytest.mark.parametrize('mask', [None,
                                  np.random.choice([True, False], size=(3, 4, 10, 10))])
def test_multi_head_attention_grad(setup_multi_head_attention_layers, mask):
    a, torch_a, layer, torch_layer = setup_multi_head_attention_layers
    torch_mask = tensor(mask, dtype=torch.bool) if mask is not None else mask

    layer(a, a, a, mask).backward()
    torch_layer(torch_a, torch_a, torch_a, torch_mask).sum().backward()

    assert_grad_close(layer.w_q, torch_layer.w_q)
    assert_grad_close(layer.w_k, torch_layer.w_k)
    assert_grad_close(layer.w_v, torch_layer.w_v)
    assert_grad_close(layer.w_o, torch_layer.w_o)
