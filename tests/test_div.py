import unittest
import numpy as np
from matterix import Tensor
import math


class TestTensorDiv(unittest.TestCase):
    def test_div(self):

        a = Tensor(0.154, requires_grad=True)
        b = Tensor(1.565, requires_grad=True)
        res = a / b

        res.backward()

        assert math.isclose(res.data.tolist(), 0.0984, rel_tol=0.01) == True
        assert math.isclose(a.grad.tolist(), 0.6390, rel_tol=0.01) == True
        assert math.isclose(b.grad.tolist(), -0.0629, rel_tol=0.01) == True

    def r_div(self):

        a = Tensor(0.154, requires_grad=True)
        res = a / 1.565

        res.backward()

        assert math.isclose(res.data.tolist(), 0.0984, rel_tol=0.01) == True
        assert math.isclose(a.grad.tolist(), 0.6390, rel_tol=0.01) == True
