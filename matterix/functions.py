from .tensor import Tensor, TensorableType, enforceTensor
from .utils import underDevelopment

# TODO: MAE, kullback leibler divergence loss

# Loss functions
def CategoricalCrossEntropy(y_train: Tensor, y_pred: Tensor) -> Tensor:
    """
    Returns the cross entropy of two different probability distribution (PD)

    Parameters
    ----------
    Arg: y_train (Tensor)
        Contains the ground truth or actual PD
    Arg: y_pred (Tensor)
        Contains the predicted values or predicted PD

    Let y_train have a distribution of `p` and y_pred have a distribution of `q`.
    Then the difference between these distribution is described by the following equation

    L(p,q) = - (p*log(q)).sum()

    Example
    --------
    nn_loss =  CategoricalCrossEntropy(y_train, y_pred)
    """
    y_train = enforceTensor(y_train)
    y_pred = enforceTensor(y_pred)

    loss_pred = -1.0 * (y_train * y_pred.log()).sum()

    return loss_pred


def MSE(y_train: Tensor, y_pred: Tensor, norm: bool = True) -> Tensor:
    """
    Returns the mean squared error for the model predictions

    Parameters
    ----------
    Arg: y_train (Tensor)
        Contains the ground truth
    Arg: y_pred (Tensor)
        Contains the predicted values

    loss = ((y_train - y_pred)**2).sum() / y_train.shape[0]

    Example
    --------
    from matterix import Functions as F
    ...

    mse_loss =  F.MSE(y_train, y_pred)
    """

    diff = y_train - y_pred

    loss = (diff * diff).sum() * (1.0 / diff.shape[0]) if norm else (diff * diff).sum()

    return loss


def RMSE(y_train: Tensor, y_pred: Tensor) -> Tensor:
    """
    Returns the root mean squared error for the model predictions

    Parameters
    ----------
    Arg: y_train (Tensor)
        Contains the ground truth
    Arg: y_pred (Tensor)
        Contains the predicted values

    sq_error = ((y_train - y_pred)**2).sum() / y_train.shape[0]
    loss = sq_error**0.5

    Example
    --------
    from matterix import Functions as F
    ...

    rmse_loss =  F.RMSE(y_train, y_pred)
    """

    diff = y_train - y_pred
    mse = ((diff * diff).sum()) * (1.0 / diff.shape[0])
    rmse = mse ** (1.0 / 2.0)

    return rmse


# Wrapper for operators
def log(x: TensorableType) -> Tensor:
    """Wrapper for the log method in Tensor"""
    x = enforceTensor(x)
    return x.log()


def logsoftmax(x: TensorableType) -> Tensor:
    """Apply log to the softmax output of a tensor"""

    x = enforceTensor(x)

    ax = x.ndim - 1
    dim = x.shape[:-1] + (1,)

    x_max = x.max(axis=ax).reshape(dim)
    exp_data = (x - x_max).exp().sum(axis=ax).log()
    output = x - x_max - exp_data.reshape(x.shape[0], 1)

    return output


def softmax(x: TensorableType) -> Tensor:
    """Wrapper for the softmax method in Tensor"""
    x = enforceTensor(x)
    return x.softmax()


def exp(x: TensorableType) -> Tensor:
    """Apply natural exp on the input tensor"""
    x = enforceTensor(x)
    return x.exp()


def sigmoid(x: TensorableType) -> Tensor:
    """Wrapper for the sigmoid method in Tensor"""
    x = enforceTensor(x)
    return x.sigmoid()


def tanh(x: TensorableType) -> Tensor:
    """Wrapper for the tanh method in Tensor"""
    x = enforceTensor(x)
    return x.tanh()


def relu(x: TensorableType) -> Tensor:
    """Wrapper for the relu method in Tensor"""
    x = enforceTensor(x)
    return x.relu()
