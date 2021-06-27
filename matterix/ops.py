from typing import Tuple
import numpy as np
from .tensor import Tensor, enforceNumpy, enforceTensor, TensorableType
from .utils import registerFn

# TODO: Slice
# TODO: Transpose


def manageBroadcasting(
    input_ndim: int, input_shape: Tuple[int], local_gradient: np.ndarray
) -> np.ndarray:
    """Handles broadcasting issue when computing gradients

    Parameters
    ----------

    Arg: input_ndim
        Rank of the tensor for which the gradient is being computed

    Arg: input_shape
        Shape of the tensor for gradient calculation

    Arg: local_gradient
        Gradient that is propogated from the output tensor.

    """

    # Given the gradient of the output is scalar there is no need for broadcasting
    if type(local_gradient) in [np.float32, float]:
        return local_gradient

    drop_dim: int = local_gradient.ndim - input_ndim

    for _ in range(drop_dim):
        local_gradient = local_gradient.sum(axis=0)

    # What is happening?
    # As we have already normalized the rank, we just sum over the dim while retaining dim
    # (2,3) + (1,3) => (2,3) :
    # (1,3) is broadcasted, so essentially we just have to sum over the _gradient along the dim which is equal to that of the child.
    for i, dim in enumerate(input_shape):
        if dim == 1:
            local_gradient = local_gradient.sum(axis=i, keepdims=True)

    return local_gradient


@registerFn(Tensor, "__add__")
def add(a: Tensor, b: Tensor) -> Tensor:
    """Returns the sum of inputs with their local gradients"""

    a = enforceTensor(a)
    b = enforceTensor(b)

    output = Tensor(a.data + b.data, requires_grad=(a.requires_grad or b.requires_grad))
    output.save_for_backward([a, b])

    def backward_fn():

        if a.requires_grad:

            a_local_gradient = output.grad.data * np.ones_like(a.data)
            a_local_gradient = manageBroadcasting(a.ndim, a.shape, a_local_gradient)

            a.grad.data += a_local_gradient

        if b.requires_grad:

            b_local_gradient = output.grad.data * np.ones_like(b.data)
            b_local_gradient = manageBroadcasting(b.ndim, b.shape, b_local_gradient)

            b.grad.data += b_local_gradient

    output.backward_fn = backward_fn

    return output


@registerFn(Tensor, "__sub__")
def sub(a: Tensor, b: Tensor) -> Tensor:
    """Returns the difference of inputs with their local gradients"""

    a = enforceTensor(a)
    b = enforceTensor(b)

    output = Tensor(a.data - b.data, requires_grad=(a.requires_grad or b.requires_grad))
    output.save_for_backward([a, b])

    def backward_fn():

        if a.requires_grad:

            a_local_gradient = output.grad.data * np.ones_like(a.data)
            a_local_gradient = manageBroadcasting(a.ndim, a.shape, a_local_gradient)

            a.grad.data += a_local_gradient

        if b.requires_grad:

            b_local_gradient = output.grad.data * -1.0 * np.ones_like(b.data)
            b_local_gradient = manageBroadcasting(b.ndim, b.shape, b_local_gradient)

            b.grad.data += b_local_gradient

    output.backward_fn = backward_fn

    return output


@registerFn(Tensor, "__mul__")
def mul(a: TensorableType, b: TensorableType) -> Tensor:
    """Returns the product of input tensor_objects with their local gradients"""

    a = enforceTensor(a)
    b = enforceTensor(b)

    output = Tensor(a.data * b.data, requires_grad=(a.requires_grad or b.requires_grad))
    output.save_for_backward([a, b])

    def backward_fn():

        if a.requires_grad:

            a_local_gradient = output.grad.data * b.data
            a_local_gradient = manageBroadcasting(a.ndim, a.shape, a_local_gradient)

            a.grad.data += a_local_gradient

        if b.requires_grad:

            b_local_gradient = output.grad.data * a.data
            b_local_gradient = manageBroadcasting(b.ndim, b.shape, b_local_gradient)

            b.grad.data += b_local_gradient

    output.backward_fn = backward_fn

    return output


@registerFn(Tensor, "__truediv__")
def div(a: TensorableType, b: TensorableType) -> Tensor:

    a = enforceTensor(a)
    b = enforceTensor(b)

    inv_b = b ** -1

    output = Tensor(
        a.data * inv_b.data, requires_grad=(a.requires_grad or inv_b.requires_grad)
    )
    output.save_for_backward([a, inv_b])

    def backward_fn():

        if a.requires_grad:

            a_local_gradient = output.grad.data * inv_b.data
            a_local_gradient = manageBroadcasting(a.ndim, a.shape, a_local_gradient)

            a.grad.data += a_local_gradient

        if inv_b.requires_grad:
            inv_b_local_gradient = output.grad.data * a.data
            inv_b_local_gradient = manageBroadcasting(
                b.ndim, b.shape, inv_b_local_gradient
            )

            inv_b.grad.data += inv_b_local_gradient

    output.backward_fn = backward_fn

    return output


@registerFn(Tensor, "sum")
def sum(a: TensorableType):

    a = enforceTensor(a)

    output = Tensor(data=a.data.sum(), requires_grad=a.requires_grad)
    output.save_for_backward([a])

    def backward_fn():

        if a.requires_grad:
            local_gradient = output.grad.data * np.ones_like(a)
            local_gradient = manageBroadcasting(a.ndim, a.shape, local_gradient)
            a.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output


@registerFn(Tensor, "__pow__")
def pow(a: TensorableType, pow: float) -> Tensor:

    a = enforceTensor(a)

    output = Tensor(a.data ** (pow), requires_grad=a.requires_grad)
    output.save_for_backward([a])

    def backward_fn():
        operation_gradient = pow * (a.data ** (pow - 1))
        local_gradient = output.grad.data * operation_gradient
        local_gradient = manageBroadcasting(a.ndim, a.shape, local_gradient)

        a.grad.data += local_gradient

    output.backward_fn = backward_fn
    return output


@registerFn(Tensor, "__matmul__")
def matmul(a: TensorableType, b: TensorableType) -> Tensor:
    """Return result of matrix multiplication of the inputs"""

    # When multiplying 1-tensor it results in an error

    a = enforceTensor(a)
    b = enforceTensor(b)

    try:
        data = a.data @ b.data
    except ValueError:
        raise TypeError(
            f"Inconsistent tensor size for the operation. {a.shape} x {b.shape} != (m,n) x (n,k)"
        )

    output = Tensor(data=data, requires_grad=(a.requires_grad or b.requires_grad))
    output.save_for_backward([a, b])

    def backward_fn():

        if a.requires_grad:
            a_local_gradient = output.grad.data @ b.data.T
            a_local_gradient = manageBroadcasting(a.ndim, a.shape, a_local_gradient)

            a.grad.data += a_local_gradient

        if b.requires_grad:

            b_local_gradient = a.data.T @ output.grad.data
            b_local_gradient = manageBroadcasting(b.ndim, b.shape, b_local_gradient)

            b.grad.data += b_local_gradient

    output.backward_fn = backward_fn

    return output


@registerFn(Tensor, "matmul")
def TensorMatMul(a: TensorableType, b: TensorableType) -> Tensor:

    return matmul(a, b)


@registerFn(Tensor, "__radd__")
def radd(a: TensorableType, b: TensorableType) -> Tensor:
    return add(b, a)


@registerFn(Tensor, "__iadd__")
def iadd(a: TensorableType, b: TensorableType) -> Tensor:

    a = enforceTensor(a)
    b = enforceTensor(b)

    a.data = a.data + b.data
    return a


@registerFn(Tensor, "__rsub__")
def rsub(a: TensorableType, b: TensorableType) -> Tensor:
    return sub(b, a)


@registerFn(Tensor, "__isub__")
def isub(a: TensorableType, b: TensorableType) -> Tensor:

    a = enforceTensor(a)
    b = enforceTensor(b)

    a.data = a.data - b.data
    return a


@registerFn(Tensor, "__rmul__")
def rmul(a: TensorableType, b: TensorableType) -> Tensor:
    return mul(b, a)


@registerFn(Tensor, "__imul__")
def imul(a: TensorableType, b: TensorableType) -> Tensor:

    a = enforceTensor(a)
    b = enforceTensor(b)

    a.data = a.data * b.data
    return a


@registerFn(Tensor, "__rtruediv__")
def rdiv(a: TensorableType, b: TensorableType) -> Tensor:
    return div(b, a)
