<p align="center">
    <img src="https://raw.githubusercontent.com/SiddeshSambasivam/MatterIx/master/assets/Logo.png?token=AKHFPP5DPO3RQLN3NBTHJGDA4DBL6" />
</p>
<p align="center">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/Matterix">
    <img alt="PyPI - License" src="https://img.shields.io/pypi/l/Matterix">
</p>

<p align="center">
  <a style="padding: 0 10px;" target="#" href="#ins">Installation</a> • 
  <a style="padding: 0 10px;" target="#" href="#releases">Releases</a> • 
  <a style="padding: 0 10px;" href="#contributing">Contributing</a> • 
  <a style="padding: 0 10px;" href="#features">Features</a>
</p>

MatterIx is a simple deep learning framework built to understand the fundamental concepts of <b>autodiff</b>, <b>optimizers</b> and <b>loss functions</b> from a first principle basis. It provide features such as automatic differentiation (autodiff), optimizers, loss functions and basic modules to create your own neural networks.

<table align="center" width="600px">
 <tr>
    <th>Feature</th>
    <th>Description</th>
    <th>Function/Specs</th>
 </tr>
 <tr>
    <td><a href="#autodiff">Autodiff</a></td>
    <td>Allows to compute gradients for tensors.</td>
    <td>First-order derivative</td>
  </tr>
  <tr>
    <td><a href="#loss">Loss functions</a></td>
    <td>Provides a metric to evaluate the model or function</td>
    <td>Mean squared error (MSE), Root mean squared error (RMSE)</td>
  </tr>
  <tr>
    <td><a href="#opt">Optimizers</a></td>
    <td>Updates the parameters of a model for a specific optimization problem</td>
    <td>Stochastic gradient descent (SGD)</td>
  </tr>
  <tr>
    <td><a href="#act">Activation functions</a></td>
    <td>It basically decides whether a neuron should be activated or not. Activation function is a non-linear transformation which applied to the output before passing it to the next layer</td>
    <td>Sigmoid, tanh, ReLU</td>
  </tr>
  <tr>
    <td><a href="#mod">Module</a></td>
    <td>Serves as a base class to design your own neural networks</td>
    <td>NIL</td>
  </tr>
</table>

<br/>

The <b>core value of matterix</b> is that it is a distilled version of pytorch so it is easier to understand what is happening under the hood.

<h3 style="font-weight:bold" id="#ins">Installation</h3>
a. Install it from github

```bash
# Install either with option-1 or option-2

# Option-1 (Preferred)
pip install git+https://github.com/SiddeshSambasivam/MatterIx.git#egg=MatterIx

# Option-2
git clone https://github.com/SiddeshSambasivam/MatterIx.git

python setup.py install

```

(or)

b. Install from PyPI

```bash
# Install directly from PyPI repository
pip install --upgrade matterix
```

<h2 style="font-weight:bold" id="features">Features</h2>

<h3 style="font-weight:bold" id="autodiff">1. Autodiff</h3>

Gradients are computed using reverse-mode autodiff. All computations are representated as a graph of tensors with each tensor holding a reference to a function which can compute the local gradient of that tensor. The calculation of the partial derivative for each tensor is completed when the entire graph is traversed.

The fundamental idea behind **`autodiff`** is that it calculates the local derivative for each variable rather than its partial derivative. This way traversing through the computational graph is simple and modular, i.e we could calculate the partial derivative of any variable with respect to the output with just one traversal, with a complexity of `O(n)`.

The difference between **partial** and **local derivative** is the way each variable is treated in each equation. When calculating the partial derivative of a function, the expression is broken down into variables, for example `c= a* b` and `d=a+b+c`, instead of using `c`, we say `a*b` in the `d= a+b+(a*b)`. On the other hand, when calculating the local derivative of a function, each element in the expression is considered a variable. I understand this might not be clear, so refer to the following <a href="https://github.com/SiddeshSambasivam/MatterIx/wiki/Understanding-reverse-mode-automatic-differentiation">explanation</a>.

<h3 style="font-weight:bold" id="loss">2. Loss functions</h3>

2.1 Mean squared error. Example

```python
from matterix.functions import MSE

y_train = ... # Actual/true value
y_pred = ... # model prediction

loss = MSE(y_train, y_pred)

```

2.2 Root Mean squared error

```python
from matterix.functions import RMSE

y_train = ... # Actual/true value
y_pred = ... # model prediction

loss = RMSE(y_train, y_pred)

```

<h3 style="font-weight:bold" id="opt">3. Optimizers</h3>

3.1 **Stochastic gradient descent**

```python
from matterix.optimizer import SGD

optimizer = SGD(model, model.parameters(), lr=0.001) # model, parameters to optimize, learning rate

# To set the gradient of the parameters to zero
optimizer.zero_grad()

# To update the parameters
optimizer.step()

```

<h3 style="font-weight:bold" id="act">4. Activation functions</h3>

**Functions:** sigmoid, tanh, relu.

All the activation functions are available from `matterix.functions`. Example,

```python
from matterix.functions import sigmoid
```

<h3 style="font-weight:bold" id="mod">5. Module</h3>

Module provides the necessary functions to design your own neural network. It has methods to set all the gradients of the parameters to zero, get all the parameters of the network.

1. Create a class which inherits from `nn.Module` to define for network
2. Initiate your parameters
3. Write a forward function

See the example below.

```python
from matterix import Tensor
import matterix.nn as nn

# To define a neural network, just inherit `Module` from `nn`
class SampleModel(nn.Module):

    def __init__(self) -> None:
        # Initilalize your parameters
        self.w1 = Tensor.randn(5, requires_grad=True)
        self.w2 = Tensor.randn(14, requires_grad=True)
        ...

    def forward(self, x) -> Tensor:

        out_1 = x @ self.w1
        ...

        return output

model = SampleModel()

model.zero_grad() # Sets the gradient of all the parameters to zero
model.parameters() # Gets all the parameters
```

<h2 style="font-weight:bold">Example</h2>

The following is a simple example

```python
# MNIST classifier

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

```

<h2 style="font-weight:bold">Development setup</h2>

Install the necessary dependecies in a seperate virtual environment

```bash
# Create a virtual environment during development to avoid dependency issues
pip install -r requirements.txt

# Before submitting a PR, run the unittests locally
pytest -v
```

<h2 style="font-weight:bold" id="releases">Release history</h2>

-   **1.1.1**

    -   **ADD:** Linear layer: Provides an abstraction to a linear model
    -   **ADD:** Log, exp and softmax functions
    -   **ADD:** Momentum to SGD
    -   **ADD:** Uniform weight initialization to linear layer
    -   **FIX:** Softmax underflow issue, Tanh bug,

-   **1.0.1**

    -   Used 1.0.0 for testing
    -   **ADD:** Tanh function, RMSE loss, randn and randint

-   **0.1.1**

    -   **ADD:** Optimizer: SGD
    -   **ADD:** Functions: Relu
    -   **ADD:** Loss functions: RMSE, MSETensor
    -   **ADD:** Module: For defining neural networks
    -   **FIX:** Floating point precision issue when calculating gradient

-   **0.1.0**

    -   First stable release
    -   **ADD:** Tensor, tensor operations, sigmoid functions
    -   **FIX:** Inaccuracies with gradient computation

<h2 style="font-weight:bold" id="contributing">Contributing</h2>

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
