import unittest

from numpy.random import randint
from matterix import Tensor
import numpy as np


class TestTensorSub(unittest.TestCase):
    def test_simple_sub(self):
        an = np.random.randint(0, 10, (1000, 1000))
        bn = np.random.randint(0, 10, (1000, 1000))
        at = Tensor(an, requires_grad=True)
        bt = Tensor(bn)
        diff_t = bt - at
        diff_t.backward(gradient=Tensor.ones_like(diff_t))

        assert diff_t.tolist() == (bn - an).tolist()
        assert at.grad.tolist() == (-1 * np.ones_like(an)).tolist()
        assert bt.grad == None

    def test_rsub(self):

        an = np.random.randint(0, 10, (10, 10))
        at = Tensor(an, requires_grad=True)
        result = 1.0 - at

        result.backward(gradient=Tensor.ones_like(result))

        assert result.tolist() == (1.0 - an).tolist()
        assert at.grad.tolist() == (-1.0 * np.ones_like(an)).tolist()

    def test_broadcast_sub(self):

        a = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)

        b = Tensor(1.0, requires_grad=True)

        c = a - b

        c.backward(gradient=Tensor.ones_like(c))

        assert a.grad.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert b.grad.tolist() == -6.0
        assert c.grad.tolist() == [[1, 1, 1], [1, 1, 1]]
