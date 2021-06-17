from typing import List, Union, NamedTuple
import numpy as np

# TODO
# 1. Check instance of the variable before any basic operations of a tensor
# 2. Backward propogation to the operations
# 3. Write unit tests for operations


class Tensor:
    def __init__(self, data=None, requires_grad: bool = False) -> None:

        data = self._checkData(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._previous = list()

    def _checkData(self, x) -> np.ndarray:
        """
        Checks the type of the data and
        raises an error when it is not a scalar or a np.array
        """

        if type(x) in (int, float):
            self.data = np.array([x], dtype=np.float32)

        elif type(x) in (np.ndarray, np.float32):
            self.data = x.astype(np.float32)

        else:
            raise ValueError(
                "Invalid data type. Input data should either be a scalar or numpy array"
            )

        return x

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, obj: "Tensor") -> "Tensor":
        def _backward():
            """
            Compute the gradients for the operator
            """
            raise NotImplementedError("Backward propogation not implemented")

        return Tensor(self.data + obj.data)

    def __sub__(self, obj: "Tensor") -> "Tensor":
        def _backward():
            """
            Compute the gradients for the operator
            """
            raise NotImplementedError("Backward propogation not implemented")

        return Tensor(self.data - obj.data)

    def __mul__(self, obj: "Tensor") -> "Tensor":
        def _backward():
            """
            Compute the gradients for the operator
            """
            raise NotImplementedError("Backward propogation not implemented")

        return Tensor(self.data * obj.data)

    def __truediv__(self, obj: "Tensor") -> "Tensor":
        def _backward():
            """
            Compute the gradients for the operator
            """
            raise NotImplementedError("Backward propogation not implemented")

        return Tensor(self.data / obj.data)

    def matmul(self, obj: "Tensor") -> "Tensor":
        if isinstance(obj, Tensor) is not True:
            raise ValueError(
                "Invalid data type. Matrix multiplication can only be performed on tensors"
            )
        return Tensor(np.matmul(self.data, obj.data))

    def transpose(self) -> "Tensor":
        """
        Return the tranpose of the tensor
        """
        return Tensor(self.data.T)

    def T(self) -> None:
        """
        Replace the transpose of the tensor in place
        """
        self.data = self.data.T

    def __repr__(self) -> str:
        return f"<Tensor(Data={self.data}, shape={self.shape}, requires_grad={self.requires_grad})>"
