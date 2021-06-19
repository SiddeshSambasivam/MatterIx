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
            res = np.array(object, dtype=np.float32)
            # if res.shape == ():
            # res.resize(1)
            return res
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
        if self.data.shape == ():
            return (1,)
        return self.data.shape

    def backward(self) -> None:
        def grad_fn(t: Tensor, grad: Tensor):

            for (child, local_gradient) in t.children:

                if child.grad is None:
                    # Addresses the case when the data is a int or a float
                    if child.data.shape == ():

                        child.grad = Tensor(np.zeros_like(1))
                    else:
                        child.grad = Tensor(np.zeros_like(child.data))

                _gradient: Tensor = local_gradient * grad

                # After a day of trying to solve this problem, I finally found this genius figuring out the same problem. Thank you Joel Grus
                # Reference: https://youtu.be/DVKaLdblCIw

                # Normalizes the rank of the tensor to that of the gradient
                if child.grad.data.ndim < _gradient.data.ndim:
                    drop_dim: int = _gradient.data.ndim - child.grad.data.ndim
                    print(f"Number of dim to drop= {drop_dim}")

                    for _ in range(drop_dim):
                        _gradient.data = _gradient.data.sum(axis=0)

                # What is happening?
                # As we have already normalized the rank, we just sum over the dim while retaining dim
                # (2,3) + (1,3) => (2,3) :
                # (1,3) is broadcasted, so essentially we just have to sum over the _gradient along the dim which is equal to that of the child.
                for i, dim in enumerate(child.data.shape):
                    if dim == 1:
                        _gradient.data = _gradient.data.sum(axis=i, keepdims=True)

                # assert (
                #     child.shape == _gradient.shape
                # ), f"Broadcasted tensor {_gradient} cannot be added to the gradient {child.grad}"

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
