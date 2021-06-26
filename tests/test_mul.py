import unittest
from matterix import Tensor
import numpy as np


class TestTensorMul(unittest.TestCase):
    def test_simple_mul(self):

        at = Tensor([2, 2, 2], requires_grad=True)
        bt = Tensor([3, 3, 3], requires_grad=True)
        mul_t = at * bt

        mul_t.backward(gradient=Tensor.ones_like(mul_t))

        assert mul_t.data.tolist() == [6, 6, 6]
        assert at.grad.data.tolist() == [3, 3, 3]
        assert bt.grad.data.tolist() == [2, 2, 2]

    def test_rmul(self):

        b = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], requires_grad=True)

        res = 2.0 * b
        res.backward(gradient=Tensor.ones_like(res))

        assert res.tolist() == [[2, 4, 6], [2, 4, 6], [2, 4, 6]]
        assert b.grad.tolist() == [[2, 2, 2], [2, 2, 2], [2, 2, 2]]

    def test_mul_grad(self):

        x = Tensor([10, -10, 10, -5, 6, 3, 1], requires_grad=True)

        sum_of_sqr = x * x

        sum_of_sqr.backward(gradient=Tensor.ones_like(sum_of_sqr))

        assert sum_of_sqr.tolist() == [100.0, 100.0, 100.0, 25.0, 36.0, 9.0, 1.0]
        assert x.grad.tolist() == [20.0, -20.0, 20.0, -10.0, 12.0, 6.0, 2.0]
