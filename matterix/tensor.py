from typing import List, Union, NamedTuple
import numpy as np

# Reference: https://pytorch.org/docs/stable/notes/autograd.html#
# It is pretty clear that there needs to be some base class which acts as a fundamental unit for all the operations
# We can think of the unit which holds the data as Tensor and the unit which performs some operation as a function
# That gives us two base classes: Tensor && Function

# Mind blocking questions:
# What happens when the value of the data is altered? How do we check that?

# arrayable are all the types which could be converted to numpy arrays
arrayable = Union[int, float, list, np.ndarray]


def parseData(data):

    # enforce only int, float data inside the numpy array -> Taken care by numpy

    if data is None:
        return None

    # if not isinstance(data, arrayable.__args__):
    #    raise TypeError('Only list, int, float or numpy array are supported')

    if not isinstance(data, np.ndarray):
        try:
            return np.array(data, dtype=np.float32)
        except ValueError:
            raise TypeError("Only list, int, float or numpy array are supported")

    return data


def checkTensor(t: "Tensor"):

    if isinstance(t, Tensor):
        return True

    return False


def checkInstance(obj: Union[arrayable, "Tensor"], error_status: str = None):
    if not isinstance(t, "Tensor"):
        try:
            obj = Tensor(parseData(obj))
        except TypeError:
            raise TypeError(error_status)

    return obj


class Tensor:
    def __init__(
        self,
        data: arrayable = None,
        dep: List["Tensor"] = [],
        requires_grad: bool = False,
    ) -> None:

        self.data = parseData(data)
        self.requires_grad = requires_grad
        self._backward = lambda: None

        if requires_grad:
            self.zero_grad()
        else:
            self.grad = None

        self._dependecies = dep

    @property
    def shape(self) -> tuple:
        if self.data is None:
            return (0,)
        return self.data.shape

    def __repr__(self) -> str:

        return f"<Tensor({self.data}, shape={self.shape})>"
