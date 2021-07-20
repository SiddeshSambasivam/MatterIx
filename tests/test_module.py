import unittest
import matterix.nn as nn
from matterix import Tensor
import matterix.functions as F


class TestModule(unittest.TestCase):
    def test_module(self):

        with self.assertRaises(TypeError):
            _ = nn.Module()

    def test_forward_implementation_error(self):

        with self.assertRaises(TypeError):

            class linear(nn.Module):
                def __init__(self):
                    super(linear, self).__init__()
                    self.linear = nn.Linear(1, 1)

            _ = linear()

    def test_parameters(self):
        class MnistModel(nn.Module):
            def __init__(self) -> None:

                self.l1 = nn.Linear(28 * 28, 128, bias=False)
                self.l2 = nn.Linear(128, 10, bias=False)

            def forward(self, x) -> Tensor:

                o1 = self.l1(x)
                o2 = self.l2(o1)
                out = F.softmax(o2)

                return out

        model = MnistModel()

        assert isinstance(model.parameters(), dict) == True
