import math

import pytest

from lm_eval.tasks.cabbq import utils as cabbq_utils
from lm_eval.tasks.esbbq import utils as esbbq_utils


# `cabbq/utils.py` and `esbbq/utils.py` are identical copies, so every case
# below runs against both.
UTILS = [
    pytest.param(cabbq_utils, id="cabbq"),
    pytest.param(esbbq_utils, id="esbbq"),
]


@pytest.mark.parametrize("utils", UTILS)
def test_acc_ambig_agg_computes_accuracy(utils):
    # (correct_and_ambig, is_ambig): two ambiguous docs, one correct, plus one
    # disambiguated doc that contributes nothing to the ambiguous accuracy.
    assert utils.acc_ambig_agg([(1, 1), (0, 1), (0, 0)]) == pytest.approx(0.5)


@pytest.mark.parametrize("utils", UTILS)
def test_acc_disambig_agg_computes_accuracy(utils):
    assert utils.acc_disambig_agg([(1, 1), (1, 1), (0, 1), (0, 0)]) == pytest.approx(
        2 / 3
    )


@pytest.mark.parametrize("utils", UTILS)
def test_acc_ambig_agg_without_ambiguous_docs_is_nan(utils):
    # E.g. `--limit 1` selecting a single disambiguated doc. The bias-score
    # aggregations already return NaN for this case; the accuracy ones used to
    # raise ZeroDivisionError and abort the whole evaluation run.
    assert math.isnan(utils.acc_ambig_agg([(0, 0), (0, 0)]))


@pytest.mark.parametrize("utils", UTILS)
def test_acc_disambig_agg_without_disambiguated_docs_is_nan(utils):
    assert math.isnan(utils.acc_disambig_agg([(0, 0), (0, 0)]))


@pytest.mark.parametrize("utils", UTILS)
def test_bias_score_aggs_without_matching_docs_are_nan(utils):
    # Pins the existing behaviour the accuracy aggregations now match.
    assert math.isnan(utils.bias_score_ambig_agg([(0, 0, 0)]))
    assert math.isnan(utils.bias_score_disambig_agg([(0, 0, 0, 0)]))
