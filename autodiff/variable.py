import numpy as np

from autodiff.grad_functions import Index, Add, Divide, Power, Mult, MatMul, Sub, Transpose, Negative


class Variable:
    def __init__(self, value: np.ndarray, fn=None, requires_grad=False):
        self.value = value
        self.requires_grad = requires_grad
        self.grad = None
        self.fn = fn

    # Allows to work with variables as if they were arrays
    def __getitem__(self, item):
        return getitem(self, item)

    def __len__(self):
        return self.value.__len__()

    def __iter__(self):
        return self.value.__iter__()

    # Implement operations
    def __add__(self, other):
        return add(self, other)

    def __truediv__(self, other):
        return truediv(self, other)

    def __pow__(self, power, modulo=None):
        return pow(self, power, modulo)

    def __mul__(self, other):
        return mul(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __neg__(self):
        return neg(self)

    def __repr__(self):
        repr = 'Variable('
        indent = ' ' * len(repr)
        value_repr = str(self.value).replace('\n', '\n' + indent)
        repr += f'{value_repr}\n'
        grad_fn_name = None if self.fn is None else self.fn.__class__.__name__
        repr += f'{indent}grad_fn: {grad_fn_name})'
        return repr

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.value == other.value
        return False

    @property
    def shape(self):
        return self.value.shape

    @property
    def T(self):
        return transpose(self)

    def backward(self, seed=None):
        if self.requires_grad:
            if seed is None:
                seed = np.ones_like(self.value)
            self.grad = seed if self.grad is None else self.grad + seed
            if self.fn:
                self.fn.backward(seed)


# Implement operations

def any_requires_grad(*variables):
    return any([v.requires_grad for v in variables])


def add(a: Variable, b: Variable):
    y = Variable(a.value + b.value, Add(a, b), any_requires_grad(a, b))
    return y


def truediv(a: Variable, b: Variable):
    y = Variable(a.value / b.value, Divide(a, b), any_requires_grad(a, b))
    return y


def pow(a, power, modulo=None):
    y = Variable(a.value ** power, Power(a, power), a.requires_grad)
    return y


def mul(a: Variable, b: Variable):
    y = Variable(a.value * b.value, Mult(a, b), any_requires_grad(a, b))
    return y


def matmul(a: Variable, b: Variable):
    y = Variable(a.value @ b.value, MatMul(a, b), any_requires_grad(a, b))
    return y


def sub(a: Variable, b: Variable):
    y = Variable(a.value - b.value, Sub(a, b), any_requires_grad(a, b))
    return y


def neg(a: Variable):
    return Variable(-a.value, Negative(a), a.requires_grad)


def transpose(a: Variable):
    return Variable(a.value.T, Transpose(a), a.requires_grad)


def getitem(a: Variable, item):
    return Variable(a.value[item], Index(a, item), a.requires_grad)

# x = Variable(np.random.rand(1, 10, 10), requires_grad=True)
# k = Variable(np.random.rand(2, 1, 8, 8), requires_grad=True)
# k1 = Variable(np.random.rand(1, 2, 3, 3), requires_grad=True)
# # print(x)
# w = conv2d(x, k)
# w = conv2d(w, k1)
# # print(w)
#
# w.backward()
# print(x.grad)
#
# x = torch.tensor(x.value, requires_grad=True, dtype=torch.float)
# k = torch.tensor(k.value, requires_grad=True, dtype=torch.float)
# k1 = torch.tensor(k1.value, requires_grad=True, dtype=torch.float)
#
# # print(x)
# w = torch.nn.functional.conv2d(x, k)
# w = torch.nn.functional.conv2d(w, k1)
# # print(w)
#
# w.sum().backward(retain_graph=True)
# print(x.grad)
