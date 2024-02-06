import pytest
import torch

from lm_eval.models.neuron_optimum import wrap_constant_batch_size


def test_wrap_constant_batch_size():
    class Tester:
        def __init__(self, batch_size):
            self.batch_size = batch_size

        @wrap_constant_batch_size
        def test_constant_batch_size(self, inputs):
            assert len(inputs) == self.batch_size
            return inputs

    batch_size_test = 8
    for i in range(1, batch_size_test + 1):
        tensor = torch.ones([i, 2, 2])
        out = Tester(batch_size=batch_size_test).test_constant_batch_size(tensor)
        torch.testing.assert_allclose(out, tensor)

    with pytest.raises(ValueError):
        Tester(batch_size=batch_size_test).test_constant_batch_size(
            torch.ones([batch_size_test + 1, 2, 2])
        )
