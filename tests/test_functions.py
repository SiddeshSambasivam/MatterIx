import unittest
from unittest import result
import numpy as np
from numpy.lib import gradient
from matterix import Tensor
import matterix.functions as F


class TestTensorFunctions(unittest.TestCase):
    def test_simple_sigmoid(self):

        an = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        at = Tensor(an)

        result = F.sigmoid(at)
        assert np.allclose(result.data, (1 / (1 + np.exp(-an))), rtol=1e-04) == True

    def test_sigmoid_gradient(self):

        an = np.array([1, 2, 3, 4])
        bn = np.array([5, 6, 7, 8])

        at = Tensor(an, requires_grad=True)
        bt = Tensor(bn, requires_grad=True)

        ct = at * bt

        dt = F.sigmoid(ct)

        dt.backward(gradient=Tensor.ones_like(dt))

        print(at.grad, "\n", bt.grad, "\n", ct.grad, "\n", dt.grad)
        assert (
            np.allclose(
                dt.data, np.array([0.9933, 1.0000, 1.0000, 1.0000]).astype(np.float32)
            )
            == True
        )
        assert (
            np.allclose(
                at.grad.data,
                np.array([0.03324028335395016, 0.000036864, 0.0000, 0.0000]),
                rtol=1e-05,
            )
            == True
        )
        assert (
            np.allclose(
                bt.grad.data,
                np.array([0.006648056670790033, 0.000012288, 0.0000, 0.0000]),
                rtol=1e-05,
            )
            == True
        )
