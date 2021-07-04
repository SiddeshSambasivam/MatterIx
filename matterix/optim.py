# TODO: Adam, RMSProp
from collections import defaultdict


class SGD:
    # TODO: Momentum, mini-batching
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

    def __init__(
        self, parameters: dict, lr: float = 0.001, momentum: float = 0.9
    ) -> None:
        self.params = parameters
        self.momentum = momentum
        self.lr = lr

        self.momentum_dict = {param: 0.0 for param, _ in self.params.items()}

    def step(self) -> None:
        """Updates the parameters of the model"""

        for param, weight in self.params.items():

            v = (self.momentum * self.momentum_dict[param]) + (weight.grad * self.lr)
            weight -= v

            self.params[param] = weight
            self.momentum_dict[param] = v

    def zero_grad(self) -> None:
        """Sets the gradients of all the parameters to zero"""

        for _, param in self.params.items():
            param.zero_grad()
