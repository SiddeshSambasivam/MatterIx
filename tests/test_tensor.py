import unittest
import numpy as np
from matterix import Tensor
from matterix.utils import InputError, RequiresGradError


class TestTensorAssignment(unittest.TestCase):
    """Unit tests for Tensor module"""

    def test_char_error(self):
        def _testCharError():
            a = Tensor(["f"])

        with self.assertRaises(ValueError):
            _testCharError()

    def test_to_list(self):
        assert Tensor([1, 2, 3]).data.tolist() == [1, 2, 3]

    def test_none_type(self):
        def _testNoneError():
            a = Tensor()

        with self.assertRaises(InputError):
            _testNoneError()

    def test_pass_tensor(self):
        def _testTensorError():
            a = Tensor([1, 2, 3])
            b = Tensor(a)

        with self.assertRaises(InputError):
            _testTensorError()

    def test_requires_grad(self):
        def _testRequiresGradError():
            a = Tensor([1, 2, 3])
            b = Tensor([4, 5, 6])
            c = (a * b).sum()

            c.backward()

        with self.assertRaises(RequiresGradError):
            _testRequiresGradError()

    def test_max1(self):

        a = Tensor(np.arange(1, 8), requires_grad=True)
        b = Tensor([np.arange(1, 8)], requires_grad=True)

        c = a.max() * b

        c.backward(Tensor.ones_like(c))
        assert c.tolist() == [[7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0]]
        assert a.grad.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 28.0]
        assert b.grad.tolist() == [[7, 7, 7, 7, 7, 7, 7]]

    def test_max2(self):
        # Test max function with axis as a parameter for tensor
        a = Tensor(np.arange(1, 8), requires_grad=True)
        b = Tensor([np.arange(1, 8), np.arange(11, 18)], requires_grad=True)
        c = a * b.max(axis=0)
        # print(b.max(axis=0))
        c.backward(Tensor.ones_like(c))
        assert c.tolist() == [11, 24, 39, 56, 75, 96, 119]
        assert a.grad.tolist() == [11, 12, 13, 14, 15, 16, 17]
        assert b.grad.tolist() == [[0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7]]

    def test_min(self):
        pass
