from typing import List, Tuple, Union
import numpy as np

from .utils import register_fn, not_working

# Reference: https://pytorch.org/docs/stable/notes/autograd.html#
# From the documentation, its clear that we can implement naive autograd with two base classes.
# Tensor: Fundamental unit of the framework to store data and all its properties.
# Function: It is base class to store context (inputs and outputs) for each operation to compute the gradient.

# All the types which could be converted to Tensors.
InputTypes = Union[int, float, list, np.ndarray]


class Tensor:
    """
    `Tensor` is a n-dimensional matrix to store floating-point data.

    All computations are representated as a graphs and each tensor represents a node in the graph.
    Gradients for tensors are computed using reverse-mode automatic differentiation.
    """

    def __init__(
        self,
        data: InputTypes = None,
        requires_grad: bool = False,
        children: List["Tensor"] = [],
    ) -> None:

        self.data = create_numpy_array(data)
        self.children = children
        self.grad = None
        self.backward_fn = lambda: None
        self.requires_grad = requires_grad

    def backward(self) -> None:
        """Initiates the gradient computation for the computational graph"""

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
                for child in v.children:
                    build_topo(child)
                gradient_tape.append(v)

        build_topo(self)

        self.grad = Tensor(np.ones_like(self.data))

        for v in reversed(gradient_tape):
            v.backward_fn()

    def matmul(self, a) -> "Tensor":
        return matmul(self, a)

    def item(self):
        return self.data

    def tolist(self) -> List[float]:
        """Returns tensor as a list"""
        return self.data.tolist()

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    @staticmethod
    def ones_like(array: InputTypes, dtype=None) -> "Tensor":

        np_object = create_numpy_array(array)

        return Tensor(np.ones_like(np_object, dtype=dtype))

    @staticmethod
    def eye(rows: int, columns: int) -> "Tensor":
        """Returns identity tensor

        Parameters
        ----------

        Arg: rows (int)
        Number of rows in the tensor

        Arg: columns (int)
        Number of columns in the tensor
        """
        return Tensor(np.eye(int(rows), int(columns)))

    @property
    def T(self) -> None:

        output = Tensor(self.data.T, requires_grad=self.requires_grad, children=[self])

        def backward_fn():

            local_gradient = Tensor(output.data.T)
            compute_gradient(self, local_gradient)

        output.backward_fn = backward_fn

        return output

    @property
    def shape(self) -> Tuple:
        """Returns the shape of the tensor"""

        if self.data.shape == ():
            return (1,)
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Returns the rank of the tensor"""
        return self.data.ndim

    def __isub__(self, object: "Tensor") -> None:

        [object] = create_tensors([object])
        self.data = self.data - object.data

        return self

    def __repr__(self) -> str:
        return f"<Tensor({self.data}, shape={self.shape})>"


def create_numpy_array(object: Union[InputTypes]) -> np.ndarray:
    """Checks if the object type is arrayable and returns the numpy array of the object

    Parameters
    ----------

    Arg: object (Union[List, int, float, np.ndarray])
    Input to be converted to a numpy array
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


def create_tensors(inputs: Union[List[Tensor], InputTypes]) -> List[Tensor]:
    """Creates and return a list of tensors from inputs"""
    tensor_objects: List = list()

    for _input in inputs:
        if not isinstance(_input, Tensor):
            tensor_objects.append(Tensor(create_numpy_array(_input)))
        else:
            tensor_objects.append(_input)

    return tensor_objects


def compute_gradient(
    tensor_object: Tensor,
    _gradient: Tensor,
) -> None:
    """Computes the gradient for the tensor_object given the output grad

    Parameters
    ----------

    Arg: tensor_object (Tensor)
    Tensor for which the gradient is to be computed

    Arg: local_gradient (Tensor)
    Parameter to pass the local gradient of `tensor_object` for an operation (addition, subtraction, ...)

    Arg: output_grad (Tensor)
    Gradient of the output tensor, `tensor_object` gradient for the operation is computed w.r.t the `output_grad`
    """

    if tensor_object.requires_grad:

        if tensor_object.grad is None:
            # Addresses the case when the data is a int or a float
            if tensor_object.data.shape == ():
                tensor_object.grad = Tensor(np.zeros_like(1))
            else:
                tensor_object.grad = Tensor(np.zeros_like(tensor_object.data))

        # Normalizes the rank of the tensor_object to that of the gradient
        if tensor_object.grad.ndim < _gradient.ndim:

            drop_dim: int = _gradient.ndim - tensor_object.grad.ndim
            for _ in range(drop_dim):
                _gradient.data = _gradient.data.sum(axis=0)

        # What is happening?
        # As we have already normalized the rank, we just sum over the dim while retaining dim
        # (2,3) + (1,3) => (2,3) :
        # (1,3) is broadcasted, so essentially we just have to sum over the _gradient along the dim which is equal to that of the child.
        for i, dim in enumerate(tensor_object.data.shape):
            if dim == 1:
                _gradient.data = _gradient.data.sum(axis=i, keepdims=True)

        tensor_object.grad += _gradient


@register_fn(Tensor, "__add__")
def add(a: Tensor, b: Tensor) -> Tensor:
    """Returns the sum of inputs with their local gradients"""

    a, b = create_tensors([a, b])

    output = Tensor(
        a.data + b.data,
        children=[a, b],
        requires_grad=(a.requires_grad or b.requires_grad),
    )

    def backward_fn():

        a_local_gradient = output.grad * Tensor(np.ones_like(a))
        b_local_gradient = output.grad * Tensor(np.ones_like(b))

        compute_gradient(a, a_local_gradient)
        compute_gradient(b, b_local_gradient)

    output.backward_fn = backward_fn

    return output


@register_fn(Tensor, "__radd__")
def radd(a: Union[int, float, list], b: Tensor) -> Tensor:
    return add(a, b)


@register_fn(Tensor, "__sub__")
def sub(a: Tensor, b: Tensor) -> Tensor:
    """Returns the difference of inputs with their local gradients"""

    a, b = create_tensors([a, b])

    output = Tensor(
        a.data - b.data,
        children=[a, b],
        requires_grad=(a.requires_grad or b.requires_grad),
    )

    def backward_fn():

        a_local_gradient = output.grad * Tensor(np.ones_like(a))
        b_local_gradient = output.grad * Tensor(-1 * np.ones_like(b))
        compute_gradient(a, a_local_gradient)
        compute_gradient(b, b_local_gradient)

    output.backward_fn = backward_fn

    return output


@register_fn(Tensor, "__rsub__")
def rsub(a: Union[int, float, list], b: Tensor) -> Tensor:
    return sub(a, b)


@register_fn(Tensor, "__mul__")
def mul(a: Tensor, b: Tensor) -> Tensor:
    """Returns the product of input tensor_objects with their local gradients"""

    a, b = create_tensors([a, b])

    output = Tensor(
        a.data * b.data,
        children=[a, b],
        requires_grad=(a.requires_grad or b.requires_grad),
    )

    def backward_fn():

        a_local_gradient = output.grad * b
        b_local_gradient = output.grad * a
        compute_gradient(a, a_local_gradient)
        compute_gradient(b, b_local_gradient)

    output.backward_fn = backward_fn

    return output


@register_fn(Tensor, "__rmul__")
def rmul(a: Union[List, int, float], b: Tensor) -> Tensor:
    return mul(a, b)


@register_fn(Tensor, "__pow__")
def power(a: Tensor, pow: int) -> Tensor:

    [a] = create_tensors([a])

    output = Tensor(a.data ** (pow), requires_grad=a.requires_grad, children=[a])

    def backward_fn():
        operation_gradient = Tensor(pow * a.data ** (pow - 1))
        _gradient = output.grad * operation_gradient

        compute_gradient(a, _gradient)

    output.backward_fn = backward_fn
    return output


@register_fn(Tensor, "__truediv__")
def div(a: Tensor, b: Tensor) -> Tensor:

    a, b = create_tensors([a, b])

    inv_b = power(b, -1)

    output = Tensor(
        a.data * inv_b.data,
        children=[a, inv_b],
        requires_grad=(a.requires_grad or inv_b.requires_grad),
    )

    def backward_fn():

        a_local_gradient = output.grad * inv_b
        b_local_gradient = output.grad * a

        compute_gradient(a, a_local_gradient)
        compute_gradient(inv_b, b_local_gradient)

    output.backward_fn = backward_fn

    return output


@register_fn(Tensor, "__rtruediv__")
def rdiv(a: Union[List, int, float], b: Tensor) -> Tensor:
    return div(a, b)


@register_fn(Tensor, "transpose")
def transpose(a: Tensor):
    a = create_tensors(a)

    return Tensor(a.data.T, requires_grad=a.requires_grad, children=a.children)


@register_fn(Tensor, "__matmul__")
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Return result of matrix multiplication of the inputs"""

    a, b = create_tensors([a, b])

    try:
        data = a.data @ b.data
    except ValueError:

        raise RuntimeError(
            f"Inconsistent tensor size for the operation. {a.shape} x {b.shape} != (m,n) x (n,k)"
        )

    output = Tensor(
        data=data,
        requires_grad=(a.requires_grad or b.requires_grad),
        children=[a, b],
    )

    def backward_fn():

        a_local_gradient = Tensor(data=output.grad.data @ b.data.T)
        b_local_gradient = Tensor(data=a.data.T @ output.grad.data)

        compute_gradient(a, a_local_gradient)
        compute_gradient(b, b_local_gradient)

    output.backward_fn = backward_fn

    return output


@register_fn(Tensor, "sum")
# @not_working
def sum(a: Tensor) -> float:

    # print("Problem", a)
    # if type(a) is not Tensor:
    [a] = create_tensors([a])
    # print("Not here in create tensors")

    output = Tensor(data=a.data.sum(), requires_grad=a.requires_grad, children=[a])

    def backward_fn():

        local_gradient = output.grad * Tensor(np.ones_like(a.data))
        compute_gradient(a, local_gradient)

    output.backward_fn = backward_fn

    return output
