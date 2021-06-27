from .tensor import Tensor
from .nn import Module

# TODO: Adam, RMSProp


class SGD:
    """
    Runs stochastic gradient descent

    Parameters
    ----------

    Arg: model (Module)
        model which needs to be optimized

    Arg: parameters (dict)
        dict of all the parameters which needs to be optimized in the model

    Arg: lr (float)
        Learning rate. Size of each gradient step
    """

    def __init__(self, model: Module, parameters: dict, lr: float = 0.001) -> None:
        self.model = model
        self.params = parameters
        self.lr = lr

    def step(self) -> None:
        """Updates the parameters of the model"""

        for k, v in self.params.items():
            v -= v.grad * self.lr
            self.params[k] = v

        self.model.__dict__.update(self.params)

    def zero_grad(self) -> None:
        """Sets the gradients of all the parameters to zero"""
        self.model.zero_grad()
