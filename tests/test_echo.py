import math
import pytest
from lm_eval.models.echo import echo


def test_echo():
    model = echo()
    eq = model.loglikelihood([("aaaa", "aa")])
    neq = model.loglikelihood([("aaaa", "bb")])
    assert 0.0 == pytest.approx(eq[0][0], rel=1e-3)
    assert math.isinf(neq[0][0]) and neq[0][0] < 0


if __name__ == "__main__":
    test_echo()
