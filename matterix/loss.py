from numpy import diff
from numpy.random import logseries
from .tensor import Tensor

# TODO: MAE, Binary cross-entropy, Categorical cross-entropy, kullback leibler divergence loss


def MSE(y_train: Tensor, y_pred: Tensor) -> Tensor:

    diff = y_train - y_pred
    loss = (diff * diff).sum() * (1.0 / diff.numel())

    return loss


def RMSE(y_train: Tensor, y_pred: Tensor) -> Tensor:

    diff = y_train - y_pred
    mse = ((diff * diff).sum()) * (1.0 / diff.numel())
    rmse = mse ** (1.0 / 2.0)

    return rmse
