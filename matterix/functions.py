import numpy as np
from .tensor import Tensor, TensorableType, enforceTensor

# TODO: Add docstrings (definition,parameters, example)
# TODO: MAE, Binary cross-entropy, Categorical cross-entropy, kullback leibler divergence loss
# TODO: Write tests for log

# BUG: Check for softmax axis param issue


def MSE(y_train: Tensor, y_pred: Tensor, norm: bool = True) -> Tensor:

    diff = y_train - y_pred

    loss = (diff * diff).sum() * (1.0 / diff.shape[0]) if norm else (diff * diff).sum()

    return loss


def RMSE(y_train: Tensor, y_pred: Tensor) -> Tensor:

    diff = y_train - y_pred
    mse = ((diff * diff).sum()) * (1.0 / diff.shape[0])
    rmse = mse ** (1.0 / 2.0)

    return rmse


def log(x: TensorableType) -> Tensor:

    x = enforceTensor(x)
    output = Tensor(np.log(x.data), requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def backward_fn():

        local_gradient = output.grad.data * (1.0 / x.data)
        x.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output


def softmax(x: TensorableType, axis=1) -> Tensor:
    # Apply exp to all values and divide the same with the sum of it
    """
    Softmax function suffers from numerical error hence must be stabilized against overflow and underflow.

    softmax(x)_i = exp(x)_i / sum(exp(x))

    When x_i is a large negative number, exp(x_i) will underflow and approximate it to zero.
    This results in denominator tending to infinity -> nan

    """

    x = enforceTensor(x)
    z = x.data - x.data.max(axis=1).reshape(x.shape[0], 1)
    x_exp = np.exp(z)

    output_data = x_exp / x_exp.sum(axis=1).reshape(x_exp.data.shape[0], 1)

    output = Tensor(output_data, requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def backward_fn():

        if x.requires_grad:

            local_gradient = output.grad.data * output_data * (1.0 - output_data)
            x.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output


def exp(x: TensorableType) -> Tensor:
    """Apply natural exp on the input tensor"""
    x = enforceTensor(x)
    output_data = np.exp(x.data)
    output = Tensor(output_data, requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def backward_fn():

        if x.requires_grad:

            local_gradient = output.grad.data * output_data
            x.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output


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
