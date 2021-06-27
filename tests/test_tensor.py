import unittest
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
