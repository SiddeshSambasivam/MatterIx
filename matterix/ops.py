from typing import Tuple
import numpy as np
from .tensor import Tensor, enforceTensor, TensorableType
from .utils import registerFn, underDevelopment

# TODO: implement min & mean, fix reshape
# BUG: max function's axis is not handled properly

# Support broadcasting issue in backwards
def manageBroadcasting(
    input_ndim: int, input_shape: Tuple[int], local_gradient: np.ndarray
) -> np.ndarray:
    """Handles broadcasting issue when computing gradients when the output gradient is broadcasted to the inputs.

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
    if type(local_gradient) in [np.float32, float] or input_ndim > local_gradient.ndim:
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


# Basic arithmetic operators
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


@registerFn(Tensor, "__pow__")
def pow(a: TensorableType, pow: float) -> Tensor:

    a = enforceTensor(a)

    output = Tensor(a.data ** (pow), requires_grad=a.requires_grad)
    output.save_for_backward([a])

    def backward_fn():

        if a.requires_grad:
            operation_gradient = pow * (a.data ** (pow - 1))
            local_gradient = output.grad.data * operation_gradient
            local_gradient = manageBroadcasting(a.ndim, a.shape, local_gradient)

            a.grad.data += local_gradient

    output.backward_fn = backward_fn
    return output


# Unary operators
@registerFn(Tensor, "log")
def log(x: Tensor) -> Tensor:

    output = Tensor(np.log(x.data), requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def backward_fn():

        local_gradient = output.grad.data * (1.0 / x.data)
        x.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output


@registerFn(Tensor, "exp")
def exp(x: Tensor) -> Tensor:

    output_data = np.exp(x.data)
    output = Tensor(output_data, requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def backward_fn():

        if x.requires_grad:

            local_gradient = output.grad.data * output_data
            x.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output


@registerFn(Tensor, "sigmoid")
def sigmoid(x: Tensor) -> Tensor:
    output_data = 1.0 / (1.0 + np.exp(-x.data))  # sig(x)

    output = Tensor(output_data, requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def backward_fn():

        if x.requires_grad:

            sigmoid_grad = output.data * (1 - output.data)  # sig(x) * (1-sig(x))
            local_gradient = output.grad.data * sigmoid_grad

            x.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output


@registerFn(Tensor, "tanh")
def tanh(x: Tensor) -> Tensor:

    tanh_x = np.tanh(x.data)

    output = Tensor(tanh_x, requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def backward_fn():

        if x.requires_grad:

            local_gradient = output.grad.data * (1 - (tanh_x * tanh_x))
            x.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output


@registerFn(Tensor, "relu")
def relu(x: TensorableType) -> Tensor:

    output = Tensor(np.maximum(x.data, 0), requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def backward_fn():

        if x.requires_grad:

            local_gradient = (x.data >= 0) * output.grad.data
            x.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output


@registerFn(Tensor, "softmax")
def softmax(x: Tensor) -> Tensor:
    """
    Softmax function suffers from numerical error hence must be stabilized against overflow and underflow.

    softmax(x)_i = exp(x)_i / sum(exp(x))

    When x_i is a large negative number, exp(x_i) will underflow and approximate it to zero.
    This results in denominator tending to zero -> nan

    """

    ax = x.ndim - 1
    dim = x.shape[:-1] + (1,)

    x_norm = x.data - x.data.max(axis=ax).reshape(dim)
    x_exp: np.ndarray = np.exp(x_norm)

    output_data = x_exp / x_exp.sum(axis=ax).reshape(dim)

    output = Tensor(output_data, requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def backward_fn():

        if x.requires_grad:

            local_gradient = output.grad.data * output_data * (1.0 - output_data)
            x.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output


# Reduction operators
@registerFn(Tensor, "sum")
def sum(a: TensorableType, axis: int = None):

    a = enforceTensor(a)
    sum_data = a.data.sum() if axis is None else a.data.sum(axis=axis)

    output = Tensor(data=sum_data, requires_grad=a.requires_grad)
    output.save_for_backward([a])

    def backward_fn():

        if a.requires_grad:
            local_gradient = output.grad.data * np.ones_like(a)
            local_gradient = manageBroadcasting(a.ndim, a.shape, local_gradient)
            a.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output


@registerFn(Tensor, "max")
def max(x: Tensor, axis: int = None) -> Tensor:
    """Returns the maximum value of the input tensor."""
    output = Tensor(np.max(x.data, axis=axis), requires_grad=x.requires_grad)

    def backward_fn():

        if x.requires_grad:

            local_gradient = output.grad.data * (x.data == output.data)
            x.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output


@registerFn(Tensor, "min")
@underDevelopment
def min() -> Tensor:
    raise NotImplementedError


@registerFn(Tensor, "mean")
def mean() -> Tensor:
    raise NotImplementedError


# Transform operators
@registerFn(Tensor, "reshape")
@underDevelopment
def reshape(x, *shape) -> "Tensor":

    x_data = x.data.reshape(*shape)
    output = Tensor(x_data, requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def backward_fn():

        if x.requires_grad:
            local_gradient = output.grad.data.reshape(x.shape)
            x.grad.data += local_gradient

    output.backward_fn = backward_fn
    return output


# Processing operators
@registerFn(Tensor, "__matmul__")
def matmul(a: TensorableType, b: TensorableType) -> Tensor:
    """Return result of matrix multiplication of the inputs"""

    a = enforceTensor(a)
    b = enforceTensor(b)

    if a.ndim == 0 or b.ndim == 0:
        raise RuntimeError(
            f"Inputs dimensions to matmul needs to be atleast 1D-Tensor."
        )

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

            a_local_gradient = np.dot(output.grad.data, b.data.T)
            a_local_gradient = manageBroadcasting(a.ndim, a.shape, a_local_gradient)

            a.grad.data += a_local_gradient.reshape(a.grad.shape)

        if b.requires_grad:

            b_local_gradient = np.dot(a.data.T, output.grad.data)
            b_local_gradient = manageBroadcasting(b.ndim, b.shape, b_local_gradient)

            b.grad.data += b_local_gradient.reshape(b.grad.shape)

    output.backward_fn = backward_fn

    return output


@registerFn(Tensor, "matmul")
def TensorMatMul(a: TensorableType, b: TensorableType) -> Tensor:
    return matmul(a, b)


# Assignment operators
@registerFn(Tensor, "__radd__")
def radd(a: TensorableType, b: TensorableType) -> Tensor:
    return add(b, a)


@registerFn(Tensor, "__iadd__")
def iadd(a: TensorableType, b: TensorableType) -> Tensor:

    a = enforceTensor(a)
    b = enforceTensor(b)

    a._data = a._data + b._data
    return a


@registerFn(Tensor, "__rsub__")
def rsub(a: TensorableType, b: TensorableType) -> Tensor:
    return sub(b, a)


@registerFn(Tensor, "__isub__")
def isub(a: TensorableType, b: TensorableType) -> Tensor:

    a = enforceTensor(a)
    b = enforceTensor(b)

    a._data = a._data - b._data
    a.grad = None
    return a


@registerFn(Tensor, "__rmul__")
def rmul(a: TensorableType, b: TensorableType) -> Tensor:
    return mul(b, a)


@registerFn(Tensor, "__imul__")
def imul(a: TensorableType, b: TensorableType) -> Tensor:

    a = enforceTensor(a)
    b = enforceTensor(b)

    a._data = a._data * b._data
    a.grad = None
    return a


@registerFn(Tensor, "__rtruediv__")
def rdiv(a: TensorableType, b: TensorableType) -> Tensor:
    return div(b, a)
