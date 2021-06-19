from typing import List, Union
import numpy as np

# Reference: https://pytorch.org/docs/stable/notes/autograd.html#
# From the documentation, its clear that we can implement naive autograd with two base classes.
# Tensor: Fundamental unit of the framework to store data and all its properties.
# Function: It is base class to store context (inputs and outputs) for each operation to compute the gradient.

# Arrayable are all the types which could be converted to Tensors.
Arrayable = Union[int, float, list, np.ndarray]


def parseArrayable(object: Union[Arrayable, "Tensor"]):
    """Checks if the object type is arrayable and returns the numpy array of the object

    Args:
    Object -- input provided by the user for a Tensor
    """

    if object is None:
        return None

    if not isinstance(object, np.ndarray):
        try:
            # enforce only int, float data inside the numpy array -> Taken care by numpy
            return np.array(object, dtype=np.float32)
        except ValueError:
            raise TypeError("Only list, int, float or numpy array are supported")

    return object


def checkTensor(t: "Tensor"):
    """Checks if it is a tensor"""
    if isinstance(t, Tensor):
        return True

    return False


class Tensor:
    def __init__(self, data: Arrayable = None, children=[]):

        self.data = parseArrayable(data)
        self.children = children
        self.grad = None

    def __repr__(self) -> str:
        return f"<Tensor({self.data}, shape={self.shape})>"

    def __add__(self, obj):

        assert type(obj) == Tensor, "Addition operation is only valid with tensors"

        return add(self, obj)

    def __sub__(self, obj):

        assert type(obj) == Tensor, "Subtraction operation is only valid with tensors"

        return sub(self, obj)

    def __mul__(self, obj):

        assert (
            type(obj) == Tensor
        ), "Multiplication operation is only valid with tensors"

        return mul(self, obj)

    @property
    def shape(self):
        return self.data.shape

    def backward(self) -> None:
        def grad_fn(t, grad):

            for (child, local_gradient) in t.children:

                if child.grad is None:
                    child.grad = Tensor(np.zeros_like(child.data))

                if local_gradient.data.ndim < grad.data.ndim:
                    print("Tackle broadcasting problem")
                    print(child, local_gradient, grad)
                    # print()
                    for _ in range(local_gradient.data.ndim - grad.data.ndim):
                        grad = grad.sum(axis=0)
                        print(grad)
                    print()
                _gradient = local_gradient * grad
                child.grad += _gradient

                grad_fn(child, _gradient)

        self.grad = Tensor(np.ones_like(self.data))
        grad_fn(self, Tensor(np.ones_like(self.data)))


def add(a: Tensor, b: Tensor):

    """
    Rule 1: If the two arrays differ in their number of dimensions, the shape of the one with fewer dimensions is padded with ones on its leading (left) side.
    Rule 2: If the shape of the two arrays does not match in any dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape.
    Rule 3: If in any dimension the sizes disagree and neither is equal to 1, an error is raised.
    """

    ct = Tensor(
        a.data + b.data,
        children=[(a, Tensor(np.ones_like(a))), (b, Tensor(np.ones_like(b)))],
    )

    return ct


def sub(a: Tensor, b: Tensor):

    return Tensor(
        a.data - b.data,
        children=[(a, Tensor(np.ones_like(a))), (b, Tensor(np.ones_like(a)))],
    )


def mul(a: Tensor, b: Tensor):

    return Tensor(a.data * b.data, children=[(a, Tensor(b.data)), (b, Tensor(a.data))])
