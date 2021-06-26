from typing import Tuple, Union
import numpy as np
from .tensor import Tensor, enforceTensor, compute_gradient
from .utils import registerFn


def manageBroadcasting(
    input_ndim: int, input_shape: Tuple[int], output: np.ndarray
) -> np.ndarray:
    """Handles broadcasting issue when computing gradients

    Parameters
    ----------
    Arg: input_ndim
    Rank of the tensor for which the gradient is being computed

    Arg: input_shape
    Shape of the tensor for gradient calculation

    Arg: output
    Result of some operation with input tensor
    """

    drop_dim: int = output.ndim - input_ndim

    for _ in range(drop_dim):
        output.data = output.data.sum(axis=0)

    # What is happening?
    # As we have already normalized the rank, we just sum over the dim while retaining dim
    # (2,3) + (1,3) => (2,3) :
    # (1,3) is broadcasted, so essentially we just have to sum over the _gradient along the dim which is equal to that of the child.
    for i, dim in enumerate(input_shape):
        if dim == 1:
            # try:
            output.data = output.data.sum(axis=i, keepdims=True)
            # except AttributeError:
            # pass
    return output


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


@registerFn(Tensor, "__radd__")
def radd(a: Union[int, float, list], b: Tensor) -> Tensor:
    return add(a, b)


@registerFn(Tensor, "sum")
def sum(a: Tensor):

    a = enforceTensor(a)

    output = Tensor(data=a.data.sum(), requires_grad=a.requires_grad)
    output.save_for_backward([a])

    def backward_fn():

        if a.requires_grad:
            local_gradient = output.grad.data * np.ones_like(a)
            manageBroadcasting(a.ndim, a.shape, output)

            a.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output
