from typing import List, Union
import numpy as np

from utils import register

# Reference: https://pytorch.org/docs/stable/notes/autograd.html#
# From the documentation, its clear that we can implement naive autograd with two base classes.
# Tensor: Fundamental unit of the framework to store data and all its properties.
# Function: It is base class to store context (inputs and outputs) for each operation to compute the gradient.

# Arrayable are all the types which could be converted to Tensors.
Arrayable = Union[int, float, list, np.ndarray]


class Tensor:
    def __init__(
        self, data: Arrayable = None, requires_grad: bool = False, children=[]
    ):

        self.data = parseArrayable(data)
        self.children = children
        self.grad = None
        self.backward_fn = lambda: None
        self.requires_grad = requires_grad

    def __repr__(self) -> str:
        return f"<Tensor({self.data}, shape={self.shape})>"

    @property
    def shape(self):
        if self.data.shape == ():
            return (1,)
        return self.data.shape

    def backward(self) -> None:
        """
        Creates a list of all the dependecies and iterates through each dependency to compute its local gradient.
        """

        if self.requires_grad is False:
            raise RuntimeError(
                "Tensors does not require grad. Enable requires_grad to compute gradients"
            )

        gradient_tape = list()
        visited = set()

        # This part of the topological sort is from https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                print(v, type(v))
                for child in v.children:
                    build_topo(child)
                gradient_tape.append(v)

        build_topo(self)

        self.grad = Tensor(np.ones_like(self.data))

        for v in reversed(gradient_tape):
            v.backward_fn()


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
            return res
        except ValueError:
            raise TypeError("Only list, int, float or numpy array are supported")

    return object


def compute_grad(tensor, local_gradient, output_grad):
    """Computes the gradient for the tensor given the output grad

    Args:
    local_gradient: A parameter to pass the local gradient for an operation (addition, subtraction, ...)

    """

    if tensor.requires_grad:

        if tensor.grad is None:
            # Addresses the case when the data is a int or a float
            if tensor.data.shape == ():
                tensor.grad = Tensor(np.zeros_like(1))
            else:
                tensor.grad = Tensor(np.zeros_like(tensor.data))

        _gradient = output_grad * local_gradient

        # Normalizes the rank of the tensor to that of the gradient
        if tensor.grad.data.ndim < _gradient.data.ndim:

            drop_dim: int = _gradient.data.ndim - tensor.grad.data.ndim
            print(f"Number of dim to drop= {drop_dim}")

            for _ in range(drop_dim):
                _gradient.data = _gradient.data.sum(axis=0)

        # What is happening?
        # As we have already normalized the rank, we just sum over the dim while retaining dim
        # (2,3) + (1,3) => (2,3) :
        # (1,3) is broadcasted, so essentially we just have to sum over the _gradient along the dim which is equal to that of the child.
        for i, dim in enumerate(tensor.data.shape):
            if dim == 1:
                _gradient.data = _gradient.data.sum(axis=i, keepdims=True)

        tensor.grad += _gradient


@register(Tensor, "__add__")
def add(a: Tensor, b: Tensor):
    """
    Returns the sum of input tensors with their local gradients
    """

    assert (
        type(a) == Tensor and type(b) == Tensor
    ), "Addition operation is only valid with tensors"

    output = Tensor(
        a.data + b.data,
        children=[a, b],
        requires_grad=(a.requires_grad or b.requires_grad),
    )

    def backward_fn():

        compute_grad(a, Tensor(np.ones_like(a)), output.grad)
        compute_grad(b, Tensor(np.ones_like(b)), output.grad)

    output.backward_fn = backward_fn

    return output


@register(Tensor, "__sub__")
def sub(a: Tensor, b: Tensor):
    """
    Returns the difference of input tensors with their local gradients
    """

    output = Tensor(
        a.data - b.data,
        children=[(a, Tensor(np.ones_like(a))), (b, Tensor(-1 * np.ones_like(a)))],
    )

    def backward_fn():

        compute_grad(a, Tensor(np.ones_like(a)), output.grad)
        compute_grad(b, Tensor(-1 * np.ones_like(b)), output.grad)

    output.backward_fn = backward_fn

    return output


@register(Tensor, "__mul__")
def mul(a: Tensor, b: Tensor):
    """
    Returns the product of input tensors with their local gradients
    """

    output = Tensor(a.data * b.data, children=[a, b])

    def backward_fn():

        compute_grad(a, Tensor(np.ones_like(b)), output.grad)
        compute_grad(b, Tensor(np.ones_like(a)), output.grad)

    output.backward_fn = backward_fn

    return output


def ones_like(array: Arrayable, dtype=None):

    np_object = parseArrayable(array)

    return Tensor(np.ones_like(np_object, dtype=dtype))
