import numpy as np
from sklearn import datasets
from matterix import Tensor

diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)
x = diabetes_x.astype(np.float32)
y = diabetes_y.astype(np.float32)

x_train, y_train = Tensor(diabetes_x[:400]), Tensor(diabetes_y[:400])
x_test, y_test = Tensor(diabetes_x[400:]), Tensor(diabetes_y[400:])


w1 = Tensor(np.random.randn(10, 28), requires_grad=True)
b1 = Tensor(np.random.randn(1, 28), requires_grad=True)
w2 = Tensor(np.random.randn(28, 1), requires_grad=True)
b2 = Tensor(np.random.randn(1), requires_grad=True)


def model(x):

    out_1 = (x @ w1) + b1
    output = (out_1 @ w2) + b2

    return output


for i in range(100):

    y_pred = model(x_train)
    loss = y_train - y_pred
    # print("Num: ", 1.0 / loss.shape[0])
    mse_loss = (loss ** 2).sum() * (1.0 / (loss.shape[0] * loss.shape[1]))

    mse_loss.backward()

    delta_1 = w1.grad * 0.001
    delta_2 = w2.grad * 0.001
    delta_3 = b1.grad * 0.001
    delta_4 = b2.grad * 0.001

    w1 -= delta_1
    b1 -= delta_3
    w2 -= delta_2
    b2 -= delta_4
    w1.zero_grad()
    w2.zero_grad()
    b1.zero_grad()
    b2.zero_grad()

    print(f"Epoch: {i} Loss: {mse_loss}")
