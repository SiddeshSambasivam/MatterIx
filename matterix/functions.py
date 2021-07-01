import numpy as np
from .tensor import Tensor, TensorableType, enforceTensor

# TODO: Add docstrings (definition,parameters, example)


def sigmoid(x: TensorableType) -> Tensor:

    x = enforceTensor(x)
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


def tanh(x: TensorableType) -> Tensor:

    x = enforceTensor(x)

    tanh_x = np.tanh(x.data)

    output = Tensor(tanh_x, requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def backward_fn():

        if x.requires_grad:

            local_gradient = output.grad.data * (1 - (tanh_x * tanh_x))
            x.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output


def relu(x: TensorableType) -> Tensor:

    x = enforceTensor(x)
    output = Tensor(np.maximum(x.data, 0), requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def backward_fn():

        if x.requires_grad:

            local_gradient = (x.data >= 0) * output.grad.data
            x.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output
