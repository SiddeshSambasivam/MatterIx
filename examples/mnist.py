import numpy as np
from matterix import Tensor, datasets
import matterix.nn as nn
import matterix.functions as F
from matterix.optim import SGD

from tqdm import trange

# Get the MNIST dataset
x_train, y_train, x_test, y_test = datasets.getMNIST()


class MnistModel(nn.Module):
    def __init__(self) -> None:

        self.l1 = nn.Linear(28 * 28, 128, bias=False)
        self.l2 = nn.Linear(128, 10, bias=False)

    def forward(self, x) -> Tensor:

        o1 = self.l1(x)
        o2 = self.l2(o1)
        out = F.softmax(o2)

        return out


model = MnistModel()
EPOCHS = 1000
batch_size = 1000
lr = 0.01

optimizer = SGD(parameters=model.parameters(), lr=lr, momentum=0.9)

t_bar = trange(EPOCHS)

losses = []

for epoch in t_bar:

    optimizer.zero_grad()

    # Batching
    ids = np.random.choice(60000, batch_size)
    x = Tensor(x_train[ids])
    y = Tensor(y_train[ids])

    y_pred = model(x)
    diff = y - y_pred
    loss = (diff ** 2).sum() * (1.0 / diff.shape[0])

    loss.backward()

    optimizer.step()
    losses.append(loss.data)

    t_bar.set_description("Epoch: %.0f Loss: %.8f" % (epoch, loss.data))

y_pred = model(Tensor(x_test))

acc = np.array(np.argmax(y_pred.data, axis=1) == np.argmax(y_test, axis=1)).sum()
print("Accuracy: ", acc / len(x_test))
