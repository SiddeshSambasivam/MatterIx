import math
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


# class TestTensorOps(unittest.TestCase):

#     @unittest.skip("Under development")
#     def test_grad_simple(self):
#         """Checks if gradient computation is correct"""

#         # Need to check the gradients of tensor which is broadcasted during operations

#         a = Tensor([[1, 2], [1, 2]], requires_grad=True)

#         b = Tensor([[1, 2], [1, 2]], requires_grad=True)

#         c = a + b
#         d = c * a

#         d.backward()

#         assert a.grad.tolist() == [[3, 6], [3, 6]]
#         assert b.grad.tolist() == [[1, 2], [1, 2]]
#         assert c.grad.tolist() == [[1, 2], [1, 2]]
#         assert d.grad.tolist() == [[1, 1], [1, 1]]

#     @unittest.skip("Under development")
#     def test_broadcasting(self):

#         a = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)

#         b = Tensor([[1]], requires_grad=True)

#         c = a + b
#         d = c * a

#         d.backward()

#         assert a.grad.tolist() == [[3, 5, 7], [3, 5, 7]]
#         assert b.grad.tolist() == [[12]]
#         assert c.grad.tolist() == [[1, 2, 3], [1, 2, 3]]
#         assert d.grad.tolist() == [[1, 1, 1], [1, 1, 1]]

#     @unittest.skip("Under development")
#     def test_singular(self):

#         a = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)

#         b = Tensor(1, requires_grad=True)
#         print(b)
#         c = a + b
#         d = c * a

#         d.backward()

#         assert a.grad.tolist() == [[3, 5, 7], [3, 5, 7]]
#         assert b.grad.tolist() == 12.0
#         assert c.grad.tolist() == [[1, 2, 3], [1, 2, 3]]
#         assert d.grad.tolist() == [[1, 1, 1], [1, 1, 1]]

#     @unittest.skip("Under development")
#     def test_ones_like(self):

#         a = np.array([1, 2, 3], dtype=np.float32)
#         b = np.array(8)
#         output_1 = Tensor.ones_like(a)
#         output_2 = Tensor.ones_like(b)

#         assert output_1.data.tolist() == [1.0, 1.0, 1.0]
#         assert output_2.data.tolist() == 1

#     @unittest.skip("Under development")
#     def test_div(self):

#         a = Tensor(0.154, requires_grad=True)
#         b = Tensor(1.565, requires_grad=True)
#         res = (a / b - a) * b

#         res.backward()

#         assert math.isclose(res.data.tolist(), -0.0870, rel_tol=0.01) == True
#         assert math.isclose(a.grad.tolist(), -0.5650, rel_tol=0.01) == True
#         assert math.isclose(b.grad.tolist(), -0.1540, rel_tol=0.01) == True

#     @unittest.skip("Under development")
#     def test_matmul1(self):

#         a = Tensor([[1, 2], [1, 2]], requires_grad=True)
#         b = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)

#         result = a @ b
#         result.backward()

#         assert result.tolist() == [[3, 6, 9], [3, 6, 9]]
#         assert result.grad.tolist() == [[1, 1, 1], [1, 1, 1]]
#         assert a.grad.tolist() == [[6, 6], [6, 6]]
#         assert b.grad.tolist() == [[2, 2, 2], [4, 4, 4]]

#     @unittest.skip("under development")
#     def test_matmul2(self):
#         def _testTypeError():
#             a = Tensor([1, 2, 3, 4])
#             b = Tensor([1, 2, 3, 4, 5, 6])

#             result = a @ b

#         with self.assertRaises(RuntimeError):
#             _testTypeError()

#     @unittest.skip("Under development")
#     def test_sum(self):

#         x = Tensor([10, -10, 10, -5, 6, 3, 1], requires_grad=True)
#         for i in range(4):

#             x.zero_grad()

#             sum_of_sqr = (x * x).sum()  # is a 0-tensor

#             if i == 0:
#                 assert sum_of_sqr.tolist() == 371.0

#             sum_of_sqr.backward()

#             delta_x = 0.1 * x.grad
#             x -= delta_x

#     @unittest.skip("Under development")
#     def test_mnist(self):

#         data = Tensor(np.random.randn(100, 784))
#         labels = Tensor(np.random.rand(100, 1))

#         w1 = Tensor(np.random.randn(784, 28), requires_grad=True)
#         b1 = Tensor(np.random.randn(1, 28), requires_grad=True)
#         w2 = Tensor(np.random.randn(28, 1), requires_grad=True)
#         b2 = Tensor(np.random.randn(1), requires_grad=True)

#         def model(x):

#             # do math
#             out_1 = (x @ w1) + b1
#             # print("problem in the model {out_1}")
#             output = (out_1 @ w2) + b2

#             return output

#         for i in range(2):

#             w1.grad = Tensor(np.zeros_like(w1))
#             w2.grad = Tensor(np.zeros_like(w1))
#             b1.grad = Tensor(np.zeros_like(w1))
#             b2.grad = Tensor(np.zeros_like(w1))

#             y_pred = model(data)

#             loss = (labels - y_pred).sum()
#             print(f"Epoch: {i} Loss: {loss.item()}")
#             loss.backward()

#             w1 -= w1.grad * 0.01
#             w2 -= w2.grad * 0.01
#             b1 -= b1.grad * 0.01
#             b2 -= b2.grad * 0.01
