import unittest
from matterix import Tensor
import numpy as np


class TestTensorExponents(unittest.TestCase):
    def test_simple_exp(self):

        an = np.random.randint(0, 10, (10, 10))
        at = Tensor(an, requires_grad=True)

        result = at * at
        result.backward(gradient=Tensor.ones_like(result))

        assert result.tolist() == (an ** 2).tolist()
        assert at.grad.tolist() == (2.0 * an).tolist()

    # def test_simple_grad(self):
    # pass
