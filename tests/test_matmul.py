import unittest
from matterix import Tensor
import numpy as np


class TestTensorMatMul(unittest.TestCase):
    def test_matmul1(self):

        a = Tensor([[1, 2], [1, 2]], requires_grad=True)
        b = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)

        result = a @ b
        result.backward(gradient=Tensor.ones_like(result))

        assert result.tolist() == [[3, 6, 9], [3, 6, 9]]
        assert result.grad.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert a.grad.tolist() == [[6, 6], [6, 6]]
        assert b.grad.tolist() == [[2, 2, 2], [4, 4, 4]]

    def test_matmul_zero_ndim(self):

        a = Tensor(2.0, requires_grad=True)
        b = Tensor(1.0, requires_grad=True)

        c = Tensor([3])

        with self.assertRaises(RuntimeError):
            d = a @ b

        with self.assertRaises(RuntimeError):
            d = c @ b

    def test_matmul_valid(self):
        def _testError():
            a = Tensor([1, 2, 3, 4])
            b = Tensor([1, 2, 3, 4, 5, 6])

            result = a @ b

        with self.assertRaises(TypeError):
            _testError()

    def test_matmul2(self):

        a = Tensor(np.arange(1, 9), requires_grad=True)
        b = Tensor(np.arange(11, 19), requires_grad=True)

        c = a @ b
        c.backward()

        assert c.tolist() == 564

        assert a.grad.tolist() == np.arange(11, 19).tolist()
        assert b.grad.tolist() == np.arange(1, 9).tolist()
