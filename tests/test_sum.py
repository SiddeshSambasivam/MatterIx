import unittest
import numpy as np
from matterix import Tensor


class TestTensorSum(unittest.TestCase):
    def test_simple_sum(self):

        a = Tensor([1, 2, 3, 4])
        b = a.sum()

        assert b.data.tolist() == 10.0

    def test_grad_sum(self):

        a = Tensor([1, 2, 3, 4], requires_grad=True)
        b = a.sum()

        b.backward()

        assert b.grad.tolist() == 1.0
        assert a.grad.tolist() == [1, 1, 1, 1]

        a.zero_grad()

        assert a.grad.tolist() == [0, 0, 0, 0]
