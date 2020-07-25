import types

ACTIVATION_REGISTRY = dict()

def register_activation(fn):

    assert isinstance(fn, types.FunctionType)

    if fn.__name__ in ACTIVATION_REGISTRY:
        err_msg = 'Cannpt register duplicate activation function ({})'
        raise ValueError(err_msg.format(fn.__name__))

    ACTIVATION_REGISTRY[fn.__name__] = fn
    
    return fn


