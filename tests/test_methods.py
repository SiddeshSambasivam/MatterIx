import unittest
from matterix import Tensor
import numpy as np


class TestTensorMethods(unittest.TestCase):
    def test_ones_like(self):

        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array(8)
        output_1 = Tensor.ones_like(a)
        output_2 = Tensor.ones_like(b)

        assert output_1.data.tolist() == [1.0, 1.0, 1.0]
        assert output_2.data.tolist() == 1
