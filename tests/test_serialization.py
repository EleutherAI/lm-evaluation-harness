import json

import numpy as np

from lm_eval.loggers.utils import _handle_non_serializable
from lm_eval.utils import handle_non_serializable


def test_handle_non_serializable_numpy_scalars():
    """All numpy scalar types must serialize to JSON numbers/booleans, not quoted strings.

    Regression: only np.int64/np.int32 were special-cased, so np.float32/float16,
    np.int16/int8/uint8 and np.bool_ fell through to str(o) and were written to the
    results/samples JSON as quoted strings (e.g. "0.5", "True", "5").
    """
    for handler in (handle_non_serializable, _handle_non_serializable):
        # numpy floats -> JSON numbers
        for f in (np.float16(0.5), np.float32(0.5), np.float64(0.5)):
            val = json.loads(json.dumps(f, default=handler))
            assert val == 0.5 and isinstance(val, float)

        # numpy booleans -> JSON booleans
        assert json.loads(json.dumps(np.bool_(True), default=handler)) is True
        assert json.loads(json.dumps(np.bool_(False), default=handler)) is False

        # numpy integers of every width -> JSON ints
        for i in (np.int8(5), np.int16(5), np.int32(5), np.int64(5), np.uint8(5)):
            val = json.loads(json.dumps(i, default=handler))
            assert val == 5 and isinstance(val, int)

        # sets still convert to lists
        assert json.loads(json.dumps({1, 2}, default=handler)) == [1, 2]
