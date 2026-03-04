import numpy as np

from autodiff.functions import log, mean, softmax, sum
from autodiff.variable import Variable, ones_like


def mse_loss(input: Variable, target: Variable):
    return mean((input - target) ** 2)


def bce_loss(input_proba: Variable, target: Variable):
    ones = ones_like(input_proba)
    loss = log(input_proba) * target + log(ones - input_proba) * (ones - target)
    return - mean(loss)


def cross_entropy(input_logits: Variable, target: Variable):
    p = softmax(input_logits, dim=1)
    c = input_logits.shape[0]
    y = Variable(np.eye(c)[target].T)
    return -mean(sum(y * log(p), dim=0))
