import inspect
from typing import Dict
from matterix.tensor import Tensor

# TODO: Linear, CONV, RNN, LSTM


class Module:
    """Base class to define neural networks"""

    def __init__(self) -> None:
        pass

    def parameters(self) -> Dict[str, Tensor]:
        """ """
        params = dict()

        for i in inspect.getmembers(self):
            if not i[0].startswith("_") and not callable(i[1]):
                if not inspect.ismethod(i[1]):
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
            v.zero_grad()
            params[k] = v

        self.__dict__.update(params)
