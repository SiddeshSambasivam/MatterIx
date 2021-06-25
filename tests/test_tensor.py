import math
import unittest
import numpy as np
from matterix import Tensor


class TestTensor(unittest.TestCase):
    """Unit tests for Tensor module"""

    # @unittest.skip("Under development")
    def test_assign(self):
        """Test to check if data assigning works"""

        def _testTypeError():
            a = Tensor(["f"])

        with self.assertRaises(TypeError):
            _testTypeError()

        assert Tensor().data == None

    # @unittest.skip("Under development")
    def test_addition_grad(self):
        """Test addition operation of a Tensor"""

        at = Tensor([1, 2], requires_grad=True)
        bt = Tensor([3, 4], requires_grad=True)
        sum_t = at + bt
        sum_t.backward()
        # print(sum_t)
        assert all(sum_t.data == np.array([4, 6]))
        assert sum_t.grad.data.tolist() == [1, 1]
        assert at.grad.data.tolist() == [1, 1]
        assert bt.grad.data.tolist() == [1, 1]

        assert (1 + at).data.tolist() == [2, 3]

    # @unittest.skip("Under development")
    def test_subtraction(self):
        """Test subtraction operation of a Tensor"""

        at = Tensor([1, 2], requires_grad=True)
        bt = Tensor([1, 2])
        diff_t = bt - at
        diff_t.backward()

        assert diff_t.data.tolist() == [0, 0]
        assert at.grad.data.tolist() == [-1, -1]
        assert bt.grad == None

    # @unittest.skip("Under development")
    def test_mul1(self):
        """Test multiplication operation of a Tensor"""

        at = Tensor([2, 2, 2], requires_grad=True)
        bt = Tensor([3, 3, 3], requires_grad=True)
        mul_t = at * bt

        mul_t.backward()

        assert mul_t.data.tolist() == [6, 6, 6]
        assert at.grad.data.tolist() == [3, 3, 3]
        assert bt.grad.data.tolist() == [2, 2, 2]

    def test_mul2(self):

        b = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], requires_grad=True)

        res = 2 * b
        res.backward()

        assert res.tolist() == [[2, 4, 6], [2, 4, 6], [2, 4, 6]]

    def test_mul3(self):

        x = Tensor([10, -10, 10, -5, 6, 3, 1], requires_grad=True)

        sum_of_sqr = x * x

        sum_of_sqr.backward()

        assert sum_of_sqr.tolist() == [100.0, 100.0, 100.0, 25.0, 36.0, 9.0, 1.0]
        assert x.grad.tolist() == [20.0, -20.0, 20.0, -10.0, 12.0, 6.0, 2.0]

    # @unittest.skip("Under development")
    def test_grad_simple(self):
        """Checks if gradient computation is correct"""

        # Need to check the gradients of tensor which is broadcasted during operations

        a = Tensor([[1, 2], [1, 2]], requires_grad=True)

        b = Tensor([[1, 2], [1, 2]], requires_grad=True)

        c = a + b
        d = c * a

        d.backward()

        assert a.grad.tolist() == [[3, 6], [3, 6]]
        assert b.grad.tolist() == [[1, 2], [1, 2]]
        assert c.grad.tolist() == [[1, 2], [1, 2]]
        assert d.grad.tolist() == [[1, 1], [1, 1]]

    # @unittest.skip("Under development")
    def test_broadcasting(self):

        a = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)

        b = Tensor([[1]], requires_grad=True)

        c = a + b
        d = c * a

        d.backward()

        assert a.grad.tolist() == [[3, 5, 7], [3, 5, 7]]
        assert b.grad.tolist() == [[12]]
        assert c.grad.tolist() == [[1, 2, 3], [1, 2, 3]]
        assert d.grad.tolist() == [[1, 1, 1], [1, 1, 1]]

    # @unittest.skip("Under development")
    def test_singular(self):

        a = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)

        b = Tensor(1, requires_grad=True)
        print(b)
        c = a + b
        d = c * a

        d.backward()

        assert a.grad.tolist() == [[3, 5, 7], [3, 5, 7]]
        assert b.grad.tolist() == 12.0
        assert c.grad.tolist() == [[1, 2, 3], [1, 2, 3]]
        assert d.grad.tolist() == [[1, 1, 1], [1, 1, 1]]

    # @unittest.skip("Under development")
    def test_ones_like(self):

        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array(8)
        output_1 = Tensor.ones_like(a)
        output_2 = Tensor.ones_like(b)

        assert output_1.data.tolist() == [1.0, 1.0, 1.0]
        assert output_2.data.tolist() == 1

    # @unittest.skip("Under development")
    def test_div(self):

        a = Tensor(0.154, requires_grad=True)
        b = Tensor(1.565, requires_grad=True)
        res = (a / b - a) * b

        res.backward()

        assert math.isclose(res.data.tolist(), -0.0870, rel_tol=0.01) == True
        assert math.isclose(a.grad.tolist(), -0.5650, rel_tol=0.01) == True
        assert math.isclose(b.grad.tolist(), -0.1540, rel_tol=0.01) == True

    # @unittest.skip("Under development")
    def test_matmul1(self):

        a = Tensor([[1, 2], [1, 2]], requires_grad=True)
        b = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)

        result = a @ b
        result.backward()

        assert result.tolist() == [[3, 6, 9], [3, 6, 9]]
        assert result.grad.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert a.grad.tolist() == [[6, 6], [6, 6]]
        assert b.grad.tolist() == [[2, 2, 2], [4, 4, 4]]

    def test_matmul2(self):
        def _testTypeError():
            a = Tensor([1, 2, 3, 4])
            b = Tensor([1, 2, 3, 4, 5, 6])

            result = a @ b

        with self.assertRaises(RuntimeError):
            _testTypeError()

    # @unittest.skip("Under development")
    def test_sum(self):

        x = Tensor([10, -10, 10, -5, 6, 3, 1], requires_grad=True)
        for i in range(4):

            x.zero_grad()

            sum_of_sqr = (x * x).sum()  # is a 0-tensor

            if i == 0:
                assert sum_of_sqr.tolist() == 371.0

            sum_of_sqr.backward()

            delta_x = 0.1 * x.grad
            x -= delta_x
