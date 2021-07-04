import unittest
import numpy as np
from matterix import Tensor
import matterix.functions as F

# TODO: write tests for mse, rmse


class TestTensorFunctions(unittest.TestCase):
    # @unittest.skip("Fixing the fundamental problem")
    def test_softmax(self):
        an = np.array([1, 2, 3, 4])
        bn = np.array([5, 6, 7, 8])

        a = Tensor(an, requires_grad=True)
        b = Tensor(bn, requires_grad=True)

        c = a * b * 0.2
        x_exp = F.exp(c)
        # sum_ = x_exp.data.sum()
        d = x_exp / x_exp.data.sum()
        # d = F.softmax(c)
        d.backward(gradient=Tensor.ones_like(d))

        assert (
            np.allclose(
                d.data,
                np.array(
                    [
                        0.003984156064689159,
                        0.016156552359461784,
                        0.09774145483970642,
                        0.8821178078651428,
                    ]
                ).astype(np.float32),
            )
            == True
        )

    def test_exp(self):

        an = np.array([1, 2, 3, 4])
        bn = np.array([5, 6, 7, 8])

        a = Tensor(an, requires_grad=True)
        b = Tensor(bn, requires_grad=True)

        c = a * b
        d = a * F.exp(c)
        d.backward(gradient=Tensor.ones_like(d))

        assert (
            np.allclose(
                d.data,
                np.array(
                    [148.4131622314453, 325509.59375, 3956447232.0, 315851827838976.0]
                ).astype(np.float32),
            )
            == True
        )

        assert (
            np.allclose(
                a.grad.data,
                np.array(
                    [890.4789428710938, 2115812.25, 29013946368.0, 2605777596448768.0]
                ),
                rtol=1e-05,
            )
            == True
        )

        assert (
            np.allclose(
                b.grad.data,
                np.array(
                    [148.4131622314453, 651019.1875, 11869341696.0, 1263407311355904.0]
                ),
                rtol=1e-05,
            )
            == True
        )

    def test_log(self):

        an = np.array([1, 2, 3, 4])
        bn = np.array([3, 6, 5, 8])

        at = Tensor(an, requires_grad=True)
        bt = Tensor(bn, requires_grad=True)

        ct = at * bt
        dt = F.log(ct)

        dt.backward(gradient=Tensor.ones_like(dt))
        print(dt.data)
        assert (
            np.allclose(
                dt.data,
                np.array(
                    [
                        1.0986123085021973,
                        2.4849066734313965,
                        2.70805025100708,
                        3.465735912322998,
                    ]
                ).astype(np.float32),
            )
            == True
        )

        assert (
            np.allclose(
                at.grad.data,
                np.array([1.0, 0.5, 0.3333333432674408, 0.25]),
                rtol=1e-05,
            )
            == True
        )

        assert (
            np.allclose(
                bt.grad.data,
                np.array(
                    [0.3333333432674408, 0.1666666716337204, 0.20000001788139343, 0.125]
                ),
                rtol=1e-05,
            )
            == True
        )

    def test_relu(self):

        an = np.array([1, 2, 3, 4])
        bn = np.array([-3, 6, -5, 8])

        at = Tensor(an, requires_grad=True)
        bt = Tensor(bn, requires_grad=True)

        ct = at * bt
        dt = F.relu(ct)

        dt.backward(gradient=Tensor.ones_like(dt))

        assert dt.tolist() == [0, 12, 0, 32]
        assert at.grad.tolist() == [0, 6, 0, 8]
        assert bt.grad.tolist() == [0, 2, 0, 4]

    def test_tanh(self):

        an = np.array([1, 2, 3, 4])
        bn = np.array([-3, 6, -5, 8])

        at = Tensor(an, requires_grad=True)
        bt = Tensor(bn, requires_grad=True)

        ct = at * bt
        dt = F.tanh(ct)

        dt.backward(gradient=Tensor.ones_like(dt))

        assert (
            np.allclose(
                dt.data,
                np.array([-0.9950547814369202, 1.0000, -1.0000, 1.0000]).astype(
                    np.float32
                ),
            )
            == True
        )

        assert (
            np.allclose(
                at.grad.data,
                np.array([-0.029597946, 0.0000, -0.0000, 0.0000]),
                rtol=1e-05,
            )
            == True
        )

        assert (
            np.allclose(
                bt.grad.data,
                np.array([0.0098659815, 0.0000, 0.0000, 0.0000]),
                rtol=1e-05,
            )
            == True
        )

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
