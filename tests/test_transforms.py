import unittest
import numpy as np
from numpy.lib.function_base import gradient
from matterix import Tensor


class TestTransforms(unittest.TestCase):
    @unittest.skip("Not implemented")
    def test_reshape(self):

        a = Tensor([1, 2, 3, 4, 5, 6, 7, 8], requires_grad=True)
        b = Tensor([[1, 2, 3, 4], [2, 4, 6, 8]], requires_grad=True)
        c = a @ b.reshape((8, 1))

        c.backward(gradient=Tensor.ones_like(c))

        assert c.data.tolist() == 170.0
        assert a.grad.data.tolist() == [1, 2, 3, 4, 2, 4, 6, 8]
        assert b.grad.data.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]

    @unittest.skip("Not implemented")
    def test_resize(self):
        pass

    @unittest.skip("Not implemented")
    def slice(self):
        pass

    @unittest.skip("Not implemented")
    def test_transpose(self):
        pass
