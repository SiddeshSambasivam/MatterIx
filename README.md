<p align="center">
    <img src="https://raw.githubusercontent.com/SiddeshSambasivam/MatterIx/master/assets/Logo.png?token=AKHFPP5DPO3RQLN3NBTHJGDA4DBL6" />
</p>
<p align="center">
    <img alt="GitHub Pipenv locked Python version" src="https://img.shields.io/github/pipenv/locked/python-version/SiddeshSambasivam/MatterIx">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/Matterix">
    <img alt="GitHub Pipenv locked dependency version (branch)" src="https://img.shields.io/github/pipenv/locked/dependency-version/SiddeshSambasivam/MatterIx/black/master">
    <img alt="PyPI - License" src="https://img.shields.io/pypi/l/Matterix">
</p>

<p align="center">
  <a style="padding: 0 10px;" target="#" href="#releases">Releases</a> • 
  <a style="padding: 0 10px;" href="#contributing">Contributing</a> • 
  <a style="padding: 0 10px;" href="#features">Features</a>
</p>

<p align="center">
MatterIx is a simple deep learning framework built to understand the fundamental concepts such as <b>autodiff</b>, <b>optimizers</b>, <b>loss functions</b> from a first principle method.
</p>

<h3 style="font-weight:bold">Features</h3>
MatterIx provide features such as automatic differntiation (autodiff) to compute gradients, optimizers, loss functions, basic modules to create your own neural networks. The <b>core value of matterix</b> is that it is a distilled version of pytorch so it is easier to understand what is happening under the hood.

At its core, matterix uses reverse-mode autodiff to compute gradients. All the computations are representated as a graph of tensors with each tensor holding a reference to a function which can compute the local gradient for the tensor. The calculation of the partial derivative for each node is completed when the entire graph is traversed.

<h3 style="font-weight:bold">Installation</h3>
a. Install it from github

```bash
# Install either with option-1 or option-2

# Option-1
pip install git+https://github.com/SiddeshSambasivam/MatterIx.git#egg=MatterIx

# Option-2
git clone https://github.com/SiddeshSambasivam/MatterIx.git

python setup.py install

```

b. Install from PyPI

```bash
# Install directly from PyPI repository
pip install --upgrade matterix
```

<h3 style="font-weight:bold">Example usage</h3>

```python

import numpy as np
from matterix import Tensor
import matterix.nn as nn
import matterix.functions as F
from matterix.loss import RMSE

x_train, y_train = Tensor(x[:1500]), Tensor(y[:1500])
x_test, y_test = Tensor(x[1500:]), Tensor(y[1500:])

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = Tensor(np.random.randn(1, 150), requires_grad=True)
        self.b1 = Tensor(np.random.randn(1, 150), requires_grad=True)
        self.w2 = Tensor(np.random.randn(150, 1), requires_grad=True)
        self.b2 = Tensor(np.random.randn(1), requires_grad=True)

    def forward(self, x) -> Tensor:
        out_1 = (x @ self.w1) + self.b1
        out_2 = F.sigmoid(out_1)
        output = (out_2 @ self.w2) + self.b2

        return output

model = Model()
optimizer = SGD(model, model.parameters())

EPOCHS = 100
t_bar = trange(EPOCHS)

for i in t_bar:

    optimizer.zero_grad()

    y_pred = model(x_train)

    loss = RMSE(y_train, y_pred)

    loss.backward()

    optimizer.step()
    t_bar.set_description("Epoch: %.0f Loss: %.5f" % (i, loss.data))
    t_bar.refresh()

```

Take a look at `examples` for different examples

<h3 style="font-weight:bold">Development setup</h3>

Install the necessary dependecies in a seperate virtual environment

```bash
# Create a virtual environment during development to avoid dependency issues
pip install -r requirements.txt

# Before submitting a PR, run the unittests locally
pytest -v
```

<h3 style="font-weight:bold" id="releases">Release history</h3>

-   **0.1.0**

    -   First stable release
    -   **ADD:** Tensor, tensor operations, sigmoid functions
    -   **FIX:** Inaccuracies with gradient computation

-   **0.1.1**
    -   **ADD:** Optimizer: SGD
    -   **ADD:** Functions: Relu
    -   **ADD:** Loss functions: RMSE, MSETensor
    -   **ADD:** Module: For defining neural networks
    -   **FIX:** Floating point precision issue when calculating gradient

<h3 style="font-weight:bold" id="contributing">Contributing</h3>

1. Fork it

2. Create your feature branch

    ```bash
    git checkout -b feature/new_feature
    ```

3. Commit your changes

    ```
    git commit -m 'add new feature'
    ```

4. Push to the branch

    ```
    git push origin feature/new_feature
    ```

5. Create a new pull request (PR)

---

Siddesh Sambasivam Suseela - [@ssiddesh45](https://twitter.com/ssiddesh45) - plutocrat45@gmail.com

Distributed under the MIT license. See `LICENSE` for more information.
