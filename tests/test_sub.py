import unittest
from matterix import Tensor
import numpy as np


# class TestTensorSub(unittest.TestCase):
#     def test_simple_sub(self):
#         at = Tensor([1, 2], requires_grad=True)
#         bt = Tensor([1, 2])
#         diff_t = bt - at
#         # diff_t.backward()

#         assert diff_t.data.tolist() == [0, 0]
#         # assert at.grad.data.tolist() == [-1, -1]
#         # assert bt.grad == None

#     def test_simple_grad(self):
#         pass
