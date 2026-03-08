from autodiff.variable import Variable


class SGD:
    def __init__(self, parameters: list[Variable], lr: float):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None


    def _step(self, parameter: Variable):
        if parameter.grad is not None:
            parameter.value = parameter.value - parameter.grad * self.lr

    def step(self):
        for p in self.parameters:
            self._step(p)
