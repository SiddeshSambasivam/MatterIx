from typing import Text
import unittest
from unittest.case import TestCase
from matterix import Tensor, tensor
import numpy as np


class TestTensorAdd(unittest.TestCase):
    def test_simple_add1(self):

        at = Tensor([1, 2], requires_grad=True)
        bt = Tensor([3, 4], requires_grad=True)
        sum_t = at + bt

        sum_t.backward(gradient=Tensor.ones_like(sum_t))

        assert sum_t.data.tolist() == [4, 6]
        assert sum_t.grad.data.tolist() == [1, 1]
        assert at.grad.data.tolist() == [1, 1]
        assert bt.grad.data.tolist() == [1, 1]

    def test_simple_add2(self):

        an = np.random.randint(0, 10, (1000, 1000))
        bn = np.random.randint(0, 10, (1000, 1000))

        at = Tensor(an)
        bt = Tensor(bn)

        assert (at + bt).tolist() == (an + bn).tolist()

    def test_radd(self):

        an = np.random.randint(0, 200, (100, 100))
        at = Tensor(an, requires_grad=True)
        result = 1.0 + at

        result.backward(gradient=Tensor.ones_like(result))

        assert result.tolist() == (1.0 + an).tolist()
        assert at.grad.tolist() == np.ones_like(an).tolist()

    def test_broadcast_sum(self):

        a = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)

        b = Tensor(1.0, requires_grad=True)

        c = a + b

        c.backward(gradient=Tensor.ones_like(c))

        assert a.grad.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert b.grad.tolist() == 6.0
        assert c.grad.tolist() == [[1, 1, 1], [1, 1, 1]]

    def test_type_error(self):
        def _typeErrorFn():
            at = Tensor([1, 2, 3, 4])
            sum_t = 1 + at

        with self.assertRaises(ValueError):
            _typeErrorFn()
