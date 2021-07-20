import inspect
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

from matterix.tensor import Tensor

# TODO: Conv, RNN, LSTM, GRU, ...


class Module(ABC):
    """Abstract base class to define neural networks"""

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Implement forward pass"""
        raise NotImplementedError("Forward function is not implemented")

    def parameters(self) -> Dict[str, Tensor]:
        """Returns all optimizable parameters"""
        params = dict()

        def filter_condition(x) -> bool:
            """Checks if object is a parameter"""

            if (isinstance(x, Tensor) and x.requires_grad == True) or isinstance(
                x, Module
            ):
                return True

            return False

        def add_prefix_to_keys(prefix: str, children_params: dict) -> dict:
            return {".".join([prefix, k]): v for k, v in children_params.items()}

        # only if predicate is true, it is included in the list
        for obj_name, obj in inspect.getmembers(self, predicate=filter_condition):
            if isinstance(obj, Module):
                # get all the parameters from that module and add prefix
                children_parameters = add_prefix_to_keys(obj_name, obj.parameters())
                params.update(children_parameters)
                continue

            params[obj_name] = obj

        return params

    def zero_grad(self) -> None:
        """Zeros out the gradient buffers of all optimizable parameters"""
        params = self.parameters()
        for k, v in params.items():
            v.zero_grad()
            params[k] = v

        self.__dict__.update(params)

    def __call__(self, x) -> Tensor:
        return self.forward(x)


class Linear(Module):
    """Abstraction of a linear model

    Parameters
    ----------
    Arg: input_dim (int)
        Number of input features

    Arg: output_dim (int)
        Number of output features

    Arg: bias (bool)
        Whether to include a bias term

    """

    def __init__(
        self, input_dim: int, output_dim: int = None, bias: bool = True
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dims = output_dim
        self.dims = (input_dim, output_dim)

        if output_dim is None:
            self.dims = (input_dim,)

        self.w = Tensor(
            np.random.uniform(-1, 1, size=self.dims) / np.sqrt(np.prod(self.dims)),
            requires_grad=True,
        )  # Features x hidden layers

        self.init_bias(set_bias=bias)

    def init_bias(self, set_bias: bool):
        """Initializes the bias of linear model"""
        if set_bias:
            if self.output_dim is None:
                self.b = Tensor.zeros_like(1, requires_grad=True)
            else:
                self.b = Tensor(np.zeros((1, self.output_dim)), requires_grad=True)
        else:
            self.b = Tensor(0.0)

    def forward(self, x) -> Tensor:

        output = x @ self.w + self.b

        return output
