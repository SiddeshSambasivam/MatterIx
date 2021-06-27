import numpy as np
from .tensor import Tensor, TensorableType, enforceTensor

# TODO: tanh, relu


def sigmoid(x: TensorableType):
    """Returns the sigmoid of a given tensor"""

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
