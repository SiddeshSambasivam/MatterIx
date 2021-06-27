import numpy as np
from matterix import Tensor
import matterix.functions as F

x = [[i] for i in range(1, 200)]
y = [[0] if i % 2 == 0 else [1] for i in range(1, 200)]

x_train, y_train = Tensor(x[:150]), Tensor(y[:150])
x_test, y_test = Tensor(x[150:]), Tensor(y[150:])

w1 = Tensor(np.random.randn(1, 150), requires_grad=True)
b1 = Tensor(np.random.randn(1, 150), requires_grad=True)
w2 = Tensor(np.random.randn(150, 1), requires_grad=True)


def model(x):

    out_1 = (x @ w1) + b1
    out_2 = F.sigmoid(out_1)
    output = out_2 @ w2

    return output


for i in range(100):

    y_pred = model(x_train)
    loss = y_train - y_pred

    mse_loss = (loss * loss).sum() * (1.0 / (loss.numel()))

    mse_loss.backward()

    w1 -= w1.grad * 0.001
    b1 -= b1.grad * 0.001
    w2 -= w2.grad * 0.001

    w1.zero_grad()
    w2.zero_grad()
    b1.zero_grad()

    print(f"Epoch: {i} Loss: {mse_loss.data}")
