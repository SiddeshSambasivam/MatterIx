import unittest
import numpy as np
import torch
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
            a = Tensor(["f"])

        with self.assertRaises(TypeError):
            _testTypeError()

        assert Tensor().data == None

    def test_addition(self):
        """
        Test addition operation of a Tensor
        """

        at = Tensor([1, 2])
        bt = Tensor([3, 4])
        sum_t = at + bt

        assert all(sum_t.data == np.array([4, 6]))

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

    def test_grad(self):
        """
        Check if gradient computation is correct

        """

        # Need to check the gradients of tensor which is broadcasted during operations

        a = Tensor([[1, 2, 3], [1, 2, 3]])

        b = Tensor([[1]])

        c = a + b
        d = c * a

        d.backward()

        at = torch.tensor(
            [[1, 2, 3], [1, 2, 3]], dtype=torch.float32, requires_grad=True
        )
        bt = torch.tensor([[1]], dtype=torch.float32, requires_grad=True)
        ct = at + bt
        ct.retain_grad()
        dt = ct * at
        dt.retain_grad()

        dt.backward(gradient=torch.ones_like(dt))
        print()
        print(b.grad, bt.grad)
        print()
        assert np.array_equal(at.grad.numpy(), a.grad.data) == True
        assert np.array_equal(bt.grad.numpy(), b.grad.data) == True
        assert np.array_equal(ct.grad.numpy(), c.grad.data) == True
        assert np.array_equal(dt.grad.numpy(), d.grad.data) == True
