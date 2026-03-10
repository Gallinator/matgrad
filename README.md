# MatGrad

[![Tests](https://github.com/Gallinator/matgrad/actions/workflows/tests.yml/badge.svg)](https://github.com/Gallinator/matgrad/actions/workflows/tests.yml)

MatGrad is a simple matrix level automatic differentiation engine written in pure python using numpy only. Inspired by [micrograd](https://github.com/karpathy/micrograd) and tested against Pytorch.

This is not a full autograd engine as it supports only a limited number of operators.
### Features
- Pytorch-like API
- Supports matrix and batched automatic differentiation
- Common operators
- Common neural network modules such as linear layers and attention
- SGD optimizer
- Common losses
- `O(n)` time computational graph traversal
- Neural networks examples

### Quick start
Clone the repo

```bash
git clone https://github.com/Gallinator/matgrad.git
```

Install requirements (it is recommended to use a conda environment)

```bash
pip install -r requirements.txt
```

To run the tests [Pytorch](https://pytorch.org/get-started/locally/) is needed.

Implement some basic operation and calculate the gradients.

```python
a = Variable([0, 1, 2], requires_grad=True)
b = Variable([3, 4, 5])
c = (a + b) ** 2

c.backward()

print(a.grad)  # [ 6 10 14]
print(b.grad)  # None

```

### Examples
The `examples` folder contains some examples on how to use the engine to train NN models on synthetic data.

|Example|Model|Task|
|-------|-----|----|
|mlp_regression.ipynb|Multi Layer Perceptron|Approximate the sine function|
|mlp_classification.ipynb|Multi Layer Perceptron|Classify 2D points taken from normal distributions|
|transformer.ipynb|Encoder-decoder transformer|Sequence reversal|

The examples are meant to be as simple as possible and do not include extensive model validation.

### Supported operations

The following `Variable` operators are implemented:

- add
- sub
- neg
- truediv
- pow
- mul
- matmul
- transpose
- slice and numpy array indexing (no assignment)
- reshape
- cat
- stack
- sin
- cos
- sigmoid
- log
- sum*
- relu
- exp
- sqrt
- softmax*
- mean*
- masked_fill

\*<small>Support along multiple dimensions</small>

The following modules are implmented:
- Linear
- LayerNorm
- Embedding
- Sequential
- Multi channel Convolution2d*
- Attention
- MultiHeadAttention

\* <small>Do not support batched inputs and is slow</small>

The following losses are available:
- Mean Squared Error
- Binary Cross Entropy
- Cross Entropy

### Implementation details
The `autodiff` directory contains the automatic differentiation engine, functions implementations, backward functions, common neural network modules and optimizers.

The engine is based on reverse accumulation and uses a computational graph to describe the operations performed on variables.

#### Forward pass

The forward pass is based on the `Variable` class which stores the data, gradient function of the operation from which it was generated and a unique id to identify it inside the computational graph.

Each operation returns a `Variable` and sets its gradient function upon creation.

Each gradient function stores the inputs and additional data such as the dimensions and masks.

#### Backward pass

When calling the backward function the following happens:
- The computational graph which starts from the `Variable` is built as a list using topological sort
- the `backward()` function is called on each variable's gradient function
- the function accumulates the gradient in the intermediate or leaf variables
- The intermediate gradients are cleared

At the end of the process the gradient of each leaf variable is available in the `.grad` attribute.
This operation has a time and space complexity of `O(n)`.


#### Gradient broadcasting
The chain rule is used to propagate the gradients from the outputs to the leaf variables.

As this engine supports batched matrix operators and is based on numpy, the gradients have to be broadcasted.
In the forward pass broadcasting occurs as per numpy:

Given two arrays $A \in \mathcal{R}^{2 \times 3 \times 4 \times 2}$ and $B \in \mathcal{R}^{3 \times 1 \times 2}$, $f(A,B)$ is a broadcasted array of shape $(2\times 3 \times 4 \times 2)$.

Therefore it can be noted that:
- $B$ gets repeated along the second dimensions $4$ times
- $B$ gets repeated along the first dimensions $2$ times
- The gradient coming from the downstream operations will be of shape $(2\times 3 \times 4 \times 2)$

This means that the gradient along those dimensions will have to be summed when differentiating with respect to $B$, because broadcasting acts as repeating the same operation multiple times using the same variable.
It is also assured that the gradient will be always broadcastable to $A$ and $B$.

Therefore in the backward pass, if the gradient from the previous function and the differentiated variable have the same shape, no broadcasting occurs.

If the number of dimensions is not the same, the gradient is accumulated along the excess dimensions starting from the left of the dimensions.

Then if the result and variable have different shapes and the same number of dimensions, the resulting gradient is accumulated along the dimensions which are not equal.

#### Tests
All the gradient functions and modules are tested against pytorch implementations. The forward pass of the most important functions and modules are also tested.
