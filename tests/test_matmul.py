import unittest
from matterix import Tensor
import numpy as np


class TestTensorMatMul(unittest.TestCase):
    def test_matmul_simple(self):

        a = Tensor([[1, 2], [1, 2]], requires_grad=True)
        b = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)

        result = a @ b
        result.backward(gradient=Tensor.ones_like(result))

        assert result.tolist() == [[3, 6, 9], [3, 6, 9]]
        assert result.grad.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert a.grad.tolist() == [[6, 6], [6, 6]]
        assert b.grad.tolist() == [[2, 2, 2], [4, 4, 4]]

    def test_matmul_valid(self):
        def _testError():
            a = Tensor([1, 2, 3, 4])
            b = Tensor([1, 2, 3, 4, 5, 6])

            result = a @ b

        with self.assertRaises(TypeError):
            _testError()
