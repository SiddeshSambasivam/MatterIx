import unittest
import numpy as np
from ..tensor import Tensor


class TestTensor(unittest.TestCase):
    """
    Unit tests for Tensor module
    """

    def test_assign(self):
        """
        Test to check if data assigning works
        """

        def _testTypeError():
            a = Tensor(['f'])

        with self.assertRaises(TypeError):
            _testTypeError()
        
        assert Tensor().data == None

    def test_addition(self):
        """
        Test addition operation of a Tensor
        """

        at = Tensor([1,2], requires_grad=True)
        bt = Tensor([3,4])
        sum_t = at + bt

        assert all(sum_t.data == np.array([4,6]))
        assert sum_t.requires_grad == True
        
        sum_t.backward()

        assert all(sum_t.grad.data == np.array([1,1]))
        assert all(at.grad.data == np.array([1,1]))
        assert bt.grad == None

        
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
