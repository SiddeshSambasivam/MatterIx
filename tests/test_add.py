from typing import Text
import unittest
from matterix import Tensor
import numpy as np


class TestTensorAdd(unittest.TestCase):
    def test_simple_add(self):

        at = Tensor([1, 2], requires_grad=True)
        bt = Tensor([3, 4], requires_grad=True)
        sum_t = at + bt

        sum_t.backward(gradient=np.ones_like(sum_t.data))
        # print(sum_t)

        assert sum_t.data.tolist() == [4, 6]
        assert sum_t.grad.data.tolist() == [1, 1]
        assert at.grad.data.tolist() == [1, 1]
        assert bt.grad.data.tolist() == [1, 1]

        # assert (1 + at).data.tolist() == [2, 3]

    def test_type_error(self):
        def _typeErrorFn():
            at = Tensor([1, 2, 3, 4])
            sum_t = 1 + at

        with self.assertRaises(ValueError):
            _typeErrorFn()
