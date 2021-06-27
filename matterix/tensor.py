from typing import List, Tuple, Union
import numpy as np

ArrayableType = Union[float, list, np.ndarray]
TensorableType = Union[float, np.ndarray, "Tensor"]

# TODO: randn, normal, randint


def enforceTensor(_input: TensorableType) -> "Tensor":
    """Converts input to tensor. This is called whenever an operation is performed"""
    if isinstance(_input, Tensor) is True:
        return _input
    else:
        return Tensor(_input)


def enforceNumpy(_input: ArrayableType, dtype=np.float64) -> np.ndarray:
    """Converts the input to numpy array. This is called only during input validation"""

    if _input is None:
        raise TypeError("No input data provided. Tensor cannot be empty.")

    if not isinstance(_input, np.ndarray):
        if type(_input) in [list, float, np.float32, np.float64]:
            return np.array(_input, dtype=dtype)
        raise ValueError("Tensor only accepts float, list and numpy array as data.")

    _input = _input.astype(dtype)

    return _input


class Tensor:
    """
    `Tensor` is a n-dimensional matrix to store floating-point (np.float32 or np.float16) data, compute gradients and perform basic operations.


    Attributes
    ----------

    data: numpy.ndarray
        stores the floating point data as a numpy array

    ctx: List[Tensor]
        list of all the operand tensors which resulted to this tensor

    grad: Tensor
        Stores the gradient for the tensor

    backward_fn: Callable[[], None]
        reference to the function to calculate the gradient for the operand tensors

    requires_grad: bool
        enforces if gradient needs to be computed for this tensor

    """

    def __init__(self, data: ArrayableType, requires_grad: bool = False) -> None:

        self.data = enforceNumpy(data)
        self.ctx: List["Tensor"] = []
        self.grad = Tensor(np.zeros_like(self.data)) if requires_grad == True else None
        self.backward_fn = lambda: None
        self.requires_grad = requires_grad

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def save_for_backward(self, inputs: List["Tensor"]) -> None:
        """Stores the tensors used to compute `self`"""
        self.ctx += inputs

    def backward(self, gradient: "Tensor" = None) -> None:
        """Traverses through the computational graph to compute gradients

        The reverse-mode automatic differentiation is used to compute the gradient for all the tensors.
        Any computation performed is represented as a graph with all each tensor in computation as a node.

        Parameters
        ----------

        Arg: gradient (Tensor)
            Gradient of the output tensor w.r.t to itself

        """

        if self.requires_grad is False:
            raise RuntimeError(
                "Tensors does not require grad. Enable requires_grad to compute gradients"
            )

        if self.data.ndim != 0:
            # Scalar values are basically 0-tensors
            if gradient is None:
                raise ValueError(
                    "Default backward function can only be computed for scalar values. Pass `gradient` for vector outputs"
                )
            self.grad = enforceTensor(gradient)
        else:
            self.grad = Tensor.ones_like(1.0)

        gradient_tape = list()
        visited = set()

        # This part of the topological sort is from https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.ctx:
                    build_topo(child)
                gradient_tape.append(v)

        build_topo(self)

        for v in reversed(gradient_tape):
            v.backward_fn()

    def tolist(self) -> List[float]:
        """Returns tensor as a list"""
        return self.data.tolist()

    def numel(self) -> int:
        """Returns the number of elements in a tensor

        Example
        -------

        >> a = Tensor([[1,2,3], [4,5,6], [7,8,9]])
        >> a.shape
        (3,3)
        >> a.numel() # 9, as there are 9 elements in the tensor
        9
        """
        _product = 1
        for dim in self.shape:
            _product *= dim

        return _product

    @staticmethod
    def zeros_like(x: ArrayableType) -> "Tensor":
        if isinstance(x, Tensor):
            return Tensor(np.zeros_like(x.data))

        x = enforceNumpy(x)
        return Tensor(np.zeros_like(x))

    @staticmethod
    def ones_like(x: ArrayableType) -> "Tensor":
        if isinstance(x, Tensor):
            return Tensor(np.ones_like(x.data))

        x = enforceNumpy(x)
        return Tensor(np.ones_like(x))

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
    def shape(self) -> Tuple:
        """Returns the shape of the tensor"""

        if self.data.shape == ():
            return (1,)
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Returns the rank of the tensor"""
        return self.data.ndim

    def __repr__(self) -> str:
        return f"Tensor({self.data}, shape={self.shape})"
