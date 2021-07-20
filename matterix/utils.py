from functools import wraps
import numpy as np


class InputError(ValueError):
    """Custom error that is raised for invalid input to Tensor"""

    def __init__(self, _object: any, message: str) -> None:
        self.object = _object
        self.message = message
        super().__init__(message)


class RequiresGradError(RuntimeError):
    def __init__(self, _object: any, message: str) -> None:
        self.object = _object
        self.message = message
        super().__init__(message)


def registerFn(cls, fn_name):
    """Decorator to add function dynamically to a class"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, fn_name, wrapper)
        return func

    return decorator


# TODO: Revisit to implement this properly
def to_categorical(x):

    a = x.flatten()

    one_hot = np.zeros((a.size, a.max() + 1))
    rows = np.arange(a.size)

    one_hot[rows, a] = 1

    return one_hot


def underDevelopment(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise NotImplementedError("Function still under development")

    return wrapper
