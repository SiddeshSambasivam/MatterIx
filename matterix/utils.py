from functools import wraps


def register(cls, fn_name):
    """Decorator to add function dynamically to a class"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, fn_name, wrapper)
        return func

    return decorator
