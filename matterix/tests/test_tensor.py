import unittest
import numpy as np
import torch
from ..tensor import Tensor, ones_like
import pytest


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

    def test_addition_grad(self):
        """
        Test addition operation of a Tensor
        """

        at = Tensor([1, 2], requires_grad=True)
        bt = Tensor([3, 4], requires_grad=True)
        sum_t = at + bt
        sum_t.backward()
        # print(sum_t)
        assert all(sum_t.data == np.array([4, 6]))
        assert sum_t.grad.data.tolist() == [1, 1]
        assert at.grad.data.tolist() == [1, 1]
        assert bt.grad.data.tolist() == [1, 1]

    # @unittest.skip("Under development")
    def test_subtraction(self):
        """
        Test subtraction operation of a Tensor
        """

        at = Tensor([1, 2], requires_grad=True)
        bt = Tensor([1, 2])
        diff_t = bt - at
        diff_t.backward()

        assert diff_t.data.tolist() == [0, 0]
        assert at.grad.data.tolist() == [-1, -1]
        assert bt.grad == None

    # @unittest.skip("Under development")
    def test_mul(self):
        """
        Test multiplication operation of a Tensor
        """

        at = Tensor([2, 2, 2], requires_grad=True)
        bt = Tensor([3, 3, 3], requires_grad=True)
        mul_t = at * bt

        mul_t.backward()

        assert mul_t.data.tolist() == [6, 6, 6]
        assert at.grad.data.tolist() == [3, 3, 3]
        assert bt.grad.data.tolist() == [2, 2, 2]

    # @unittest.skip("Under development")
    def test_grad_simple(self):
        """
        Check if gradient computation is correct

        """

        # Need to check the gradients of tensor which is broadcasted during operations

        a = Tensor([[1, 2], [1, 2]], requires_grad=True)

        b = Tensor([[1, 2], [1, 2]], requires_grad=True)

        c = a + b
        d = c * a

        d.backward()

        at = torch.tensor([[1, 2], [1, 2]], dtype=torch.float32, requires_grad=True)
        bt = torch.tensor([[1, 2], [1, 2]], dtype=torch.float32, requires_grad=True)
        ct = at + bt
        ct.retain_grad()
        dt = ct * at
        dt.retain_grad()

        dt.backward(gradient=torch.ones_like(dt))

        assert np.array_equal(at.grad.numpy(), a.grad.data) == True
        assert np.array_equal(bt.grad.numpy(), b.grad.data) == True
        assert np.array_equal(ct.grad.numpy(), c.grad.data) == True
        assert np.array_equal(dt.grad.numpy(), d.grad.data) == True

    # @unittest.skip("Under development")
    def test_broadcasting(self):

        a = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)

        b = Tensor([[1]], requires_grad=True)

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

        assert np.array_equal(at.grad.numpy(), a.grad.data) == True
        assert np.array_equal(bt.grad.numpy(), b.grad.data) == True
        assert np.array_equal(ct.grad.numpy(), c.grad.data) == True
        assert np.array_equal(dt.grad.numpy(), d.grad.data) == True

    # @unittest.skip("Under development")
    def test_singular(self):

        a = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)

        b = Tensor(1, requires_grad=True)

        c = a + b
        d = c * a

        d.backward()

        at = torch.tensor(
            [[1, 2, 3], [1, 2, 3]], dtype=torch.float32, requires_grad=True
        )
        bt = torch.tensor(1, dtype=torch.float32, requires_grad=True)
        ct = at + bt
        ct.retain_grad()
        dt = ct * at
        dt.retain_grad()

        dt.backward(gradient=torch.ones_like(dt))

        assert np.array_equal(at.grad.numpy(), a.grad.data) == True
        assert np.array_equal(bt.grad.numpy(), b.grad.data) == True
        assert np.array_equal(ct.grad.numpy(), c.grad.data) == True
        assert np.array_equal(dt.grad.numpy(), d.grad.data) == True

    # @unittest.skip("Under development")
    def test_ones_like(self):

        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array(8)
        output_1 = ones_like(a)
        output_2 = ones_like(b)

        assert output_1.data.tolist() == [1.0, 1.0, 1.0]
        assert output_2.data.tolist() == 1
