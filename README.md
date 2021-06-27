<p align="center">
    <img src="https://raw.githubusercontent.com/SiddeshSambasivam/MatterIx/master/assets/Logo.png?token=AKHFPP5DPO3RQLN3NBTHJGDA4DBL6" />
</p>
<p align="center">
    <img alt="PyPI - License" src="https://img.shields.io/pypi/l/Matterix">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/Matterix">
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
"""
Task: Simple model to predict if a given number is odd/even
"""
import numpy as np
from matterix import Tensor
import matterix.functions as F

# Prepare training data
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
