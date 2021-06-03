import unittest
from ..tensor import Tensor


class TestTensor(unittest.TestCase):
    """
    Unit tests for Tensor module
    """

    def test_assign(self):
        """
        Test to check if data assigning works
        """

        def _test():
            a = Tensor([1])

        with self.assertRaises(ValueError):
            _test()

    def test_addition(self):
        """
        Test addition operation of a Tensor
        """

        at = Tensor(1)
        bt = Tensor(2)
        sum_t = at + bt

        assert sum_t.data == 3

    def test_subtraction(self):
        """
        Test subtraction operation of a Tensor
        """

        at = Tensor(1)
        bt = Tensor(2)
        diff_t = bt - at

        assert diff_t.data == 1

    def test_mul(self):
        """
        Test multiplication operation of a Tensor
        """

        at = Tensor(1)
        bt = Tensor(2)
        mul_t = at * bt

        assert mul_t.data == 2

    def test_div(self):
        """
        Test addition operation of a Tensor
        """

        at = Tensor(1)
        bt = Tensor(2)
        div_t = at / bt

        assert div_t.data == 0.5
