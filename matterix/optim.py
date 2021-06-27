# TODO: SGD, Adam, RMSProp

# Model (params)
# -> Optimizer (which updates the parameters)
# -> Needs to be reflected in the Model (params)


class SGD:
    def __init__(self, model, parameters, lr: float = 0.001) -> None:
        self.model = model
        self.params = parameters
        self.lr = lr

    def step(self):

        for k, v in self.params.items():
            v -= v.grad * self.lr
            self.params[k] = v
        self.model.__dict__.update(self.params)

    def zero_grad(self) -> None:
        self.model.zero_grad()
