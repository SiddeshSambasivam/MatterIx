import inspect
from typing import Dict

from matterix.tensor import Tensor

# TODO: Conv, RNN, LSTM
# TODO: Add docstrings for linear layer


class Module:
    """Base class to define neural networks"""

    def __init__(self) -> None:
        pass

    def parameters(self) -> Dict[str, Tensor]:
        """Returns all optimizable parameters"""
        params = dict()

        for i in inspect.getmembers(self):
            if not i[0].startswith("_") and not inspect.ismethod(i[1]):
                # Account for activation functions
                if callable(i[1]) and hasattr(i[1], "parameters"):

                    children_params = i[1].parameters()
                    params.update(
                        {".".join([i[0], k]): v for k, v in children_params.items()}
                    )
                else:
                    params[i[0]] = i[1]

        return params

    def __call__(self, x) -> Tensor:

        forward_fn = getattr(self, "forward", None)  # None is a default value
        if callable(forward_fn):
            return self.forward(x)
        else:
            raise NotImplementedError("Forward function is not implemented")

    def zero_grad(self) -> None:

        params = self.parameters()
        for k, v in params.items():

            # Addresses the activation function issue
            if hasattr(v, "zero_grad") is False:
                raise RuntimeError(
                    f"{v.__name__} is not a layer and hence should not be added to the model initialization"
                )

            v.zero_grad()
            params[k] = v

        self.__dict__.update(params)


class Linear(Module):
    def __init__(
        self, input_dim: int, output_dim: int = None, has_bias: bool = True
    ) -> None:
        super().__init__()

        if output_dim is None:
            dims = [input_dim]
        else:
            dims = [input_dim, output_dim]

        self.w = Tensor.randn(*dims, requires_grad=True)  # Features x hidden layers

        if has_bias:
            if output_dim is None:
                self.b = Tensor.randn(1, requires_grad=True)
            else:
                self.b = Tensor.randn(1, output_dim, requires_grad=True)
        else:
            self.b = 0

    def forward(self, x) -> Tensor:
        output = x @ self.w + self.b

        return output
