import re
import string

import numpy as np

from lm_eval.api.registry import register_metric


### the code used in the `exact_match_hf_evaluate` function is ported from
### https://github.com/huggingface/evaluate/blob/main/metrics/exact_match/exact_match.py
### which is under the apache license.

# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0


# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def exact_match_hf_evaluate(
    predictions: list[str],
    references: list[str],
    regexes_to_ignore: list[str] | None = None,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    ignore_numbers: bool = False,
    multiple_targets: bool = False,
):
    _predictions = np.asarray(predictions)
    _references = np.asarray(references)

    if regexes_to_ignore:
        for s in regexes_to_ignore:
            _predictions = np.array([re.sub(s, "", str(x)) for x in _predictions])
            _references = np.array([re.sub(s, "", str(x)) for x in _references])

    if ignore_case:
        _predictions = np.char.lower(_predictions)
        _references = np.char.lower(_references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        _predictions = np.array([str(x).translate(repl_table) for x in _predictions])
        _references = np.array([str(x).translate(repl_table) for x in _references])

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        _predictions = np.array([str(x).translate(repl_table) for x in _predictions])
        _references = np.array([str(x).translate(repl_table) for x in _references])

    score_list = _predictions == _references

    return {"exact_match": np.mean(score_list)}


@register_metric(
    metric="exact_match",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def exact_match_fn(references: list[str], predictions: list[str], **kwargs):
    return exact_match_hf_evaluate(
        predictions=predictions, references=references, **kwargs
    )
