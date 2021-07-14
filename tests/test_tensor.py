import unittest
import numpy as np
from matterix import Tensor


class TestTensorAssignment(unittest.TestCase):
    """Unit tests for Tensor module"""

    def test_char_error(self):
        def _testCharError():
            a = Tensor(["f"])

        with self.assertRaises(ValueError):
            _testCharError()

    def test_pass_list(self):

        assert Tensor([1, 2, 3]).data.tolist() == [1, 2, 3]

    def test_none_type(self):
        def _testNoneError():
            a = Tensor()

        with self.assertRaises(TypeError):
            _testNoneError()

    def test_pass_tensor(self):
        def _testTensorError():
            a = Tensor([1, 2, 3])
            b = Tensor(a)

        with self.assertRaises(ValueError):
            _testTensorError()

    def test_max(self):

        a = Tensor(np.arange(1, 8), requires_grad=True)
        b = Tensor([np.arange(1, 8)], requires_grad=True)

        c = a.max() * b

        c.backward(Tensor.ones_like(c))
        assert c.tolist() == [[7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0]]
        assert a.grad.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 28.0]
        assert b.grad.tolist() == [[7, 7, 7, 7, 7, 7, 7]]

    def test_min(self):
        pass
