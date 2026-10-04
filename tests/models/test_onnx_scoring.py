"""ONNX scoring-window regressions that do not require downloaded models."""

import json
import math

import numpy as np
import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM, CachingLM, hash_args
from lm_eval.models.onnxruntime_genai import ONNXRuntimeGenAILM
from lm_eval.models.onnxruntime_ort import ONNXRuntimeLM


TRANSITION_LOGITS = np.array(
    [
        [2, -1, 0, 1, -2, 3],
        [-1, 3, 2, -2, 0, 1],
        [0, -2, 3, 1, 2, -1],
        [1, 0, -1, 3, -2, 2],
        [3, 1, -2, 0, 2, -1],
        [-2, 2, 1, -1, 3, 0],
    ],
    dtype=np.float32,
)


def _transition_score(predecessors, targets):
    """Independent scalar oracle, without the backend's window or softmax code."""
    scores = []
    greedy = []
    for predecessor, target in zip(predecessors, targets, strict=True):
        row = [float(value) for value in TRANSITION_LOGITS[predecessor]]
        scores.append(row[target] - math.log(sum(math.exp(value) for value in row)))
        greedy.append(target == row.index(max(row)))
    return sum(scores), all(greedy)


def _make_scoring_lm(model_cls, max_length):
    lm = model_cls.__new__(model_cls)
    LM.__init__(lm)
    lm.max_length = max_length
    lm._bos_token_id = 0
    lm._eot_token_id = 0
    lm.tok_encode = lambda text: [int(token) for token in text.split()]
    lm.forward_inputs = []

    def forward(tokens):
        assert 0 < len(tokens) <= max_length
        lm.forward_inputs.append(list(tokens))
        return TRANSITION_LOGITS[tokens]

    lm._forward_logits = forward
    return lm


@pytest.mark.parametrize("model_cls", [ONNXRuntimeGenAILM, ONNXRuntimeLM])
@pytest.mark.parametrize("max_length", [1, 2, 4])
def test_loglikelihood_scores_a_full_continuation(model_cls, max_length):
    lm = _make_scoring_lm(model_cls, max_length)
    context = [0]
    continuation = [1, 2, 3, 4][:max_length]
    expected = _transition_score(context + continuation[:-1], continuation)

    (actual,) = lm._loglikelihood_tokens(
        [(None, context, continuation)], disable_tqdm=True
    )

    assert actual[0] == pytest.approx(expected[0], abs=1e-6)
    assert actual[1] == expected[1]
    assert lm.forward_inputs == [context + continuation[:-1]]


@pytest.mark.parametrize("model_cls", [ONNXRuntimeGenAILM, ONNXRuntimeLM])
@pytest.mark.parametrize(
    ("context", "continuation", "expected_input", "scored_targets"),
    [
        ([0, 1], [2], [0, 1], [2]),
        ([0, 1], [2, 3], [0, 1, 2], [2, 3]),
        ([0, 1, 2, 3, 4], [5, 1], [2, 3, 4, 5], [5, 1]),
        ([0], [1, 2, 3, 4, 5, 1], [2, 3, 4, 5], [3, 4, 5, 1]),
    ],
)
def test_loglikelihood_left_truncation(
    model_cls, context, continuation, expected_input, scored_targets
):
    lm = _make_scoring_lm(model_cls, max_length=4)
    predecessors = expected_input[-len(scored_targets) :]
    expected = _transition_score(predecessors, scored_targets)

    (actual,) = lm._loglikelihood_tokens(
        [(None, context, continuation)], disable_tqdm=True
    )

    assert actual[0] == pytest.approx(expected[0], abs=1e-6)
    assert actual[1] == expected[1]
    assert lm.forward_inputs == [expected_input]


@pytest.mark.parametrize("model_cls", [ONNXRuntimeGenAILM, ONNXRuntimeLM])
@pytest.mark.parametrize("max_length", [1, 2, 4])
@pytest.mark.parametrize("num_tokens", [0, 1, 3, 4, 5, 8, 9])
def test_rolling_loglikelihood_scores_every_token(model_cls, max_length, num_tokens):
    lm = _make_scoring_lm(model_cls, max_length)
    tokens = ([1, 2, 3, 4, 5] * 2)[:num_tokens]
    request = Instance(
        request_type="loglikelihood_rolling",
        doc={},
        arguments=(" ".join(map(str, tokens)),),
        idx=0,
    )
    expected, _ = _transition_score([0] + tokens[:-1] if tokens else [], tokens)

    (actual,) = lm.loglikelihood_rolling([request], disable_tqdm=True)

    assert actual == pytest.approx(expected, abs=2e-6)
    assert all(len(window) <= max_length for window in lm.forward_inputs)


@pytest.mark.parametrize("model_cls", [ONNXRuntimeGenAILM, ONNXRuntimeLM])
def test_empty_continuation_does_not_run_inference(model_cls):
    lm = _make_scoring_lm(model_cls, max_length=4)

    assert lm._loglikelihood_tokens([(None, [0], [])], disable_tqdm=True) == [
        (0.0, True)
    ]
    assert lm.forward_inputs == []


def test_full_window_greedy_flag_includes_first_target():
    lm = _make_scoring_lm(ONNXRuntimeLM, max_length=2)
    # 1 is not greedy after 0, but 1 is greedy after 1. Dropping the first
    # target incorrectly reports that the whole continuation is greedy.
    (result,) = lm._loglikelihood_tokens([(None, [0], [1, 1])], disable_tqdm=True)

    assert result[1] is False


@pytest.fixture
def local_onnx_lm(tmp_path):
    """Load an actual local ONNX transition graph through the production backend."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Gather", ["transitions", "input_ids"], ["logits"])],
        "transition-model",
        [
            onnx.helper.make_tensor_value_info(
                "input_ids", onnx.TensorProto.INT64, [1, "sequence"]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                "logits", onnx.TensorProto.FLOAT, [1, "sequence", 6]
            )
        ],
        [onnx.numpy_helper.from_array(TRANSITION_LOGITS, "transitions")],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=10
    )
    onnx.checker.check_model(model)
    onnx.save(model, tmp_path / "model.onnx")
    (tmp_path / "genai_config.json").write_text(
        json.dumps(
            {"model": {"context_length": 4, "bos_token_id": 0, "eos_token_id": 0}}
        )
    )

    prefix_token = str(0)
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {str(i): i for i in range(6)}, unk_token=prefix_token
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=prefix_token,
        eos_token=prefix_token,
        unk_token=prefix_token,
    ).save_pretrained(tmp_path)

    return ONNXRuntimeLM(pretrained=str(tmp_path), execution_provider="cpu")


def test_real_onnx_full_continuation(local_onnx_lm):
    request = Instance(
        request_type="loglikelihood",
        doc={},
        arguments=("0", " 1 2 3 4"),
        idx=0,
    )
    expected = _transition_score([0, 1, 2, 3], [1, 2, 3, 4])

    (actual,) = local_onnx_lm.loglikelihood([request], disable_tqdm=True)

    assert actual[0] == pytest.approx(expected[0], abs=1e-6)
    assert actual[1] == expected[1]


@pytest.mark.parametrize("continuation", ["1 2 3 4", "0 1 2 3 4"])
def test_real_onnx_empty_context_uses_prefix(local_onnx_lm, continuation):
    request = Instance(
        request_type="loglikelihood",
        doc={},
        arguments=("", continuation),
        idx=0,
    )
    # An explicit leading BOS is reused rather than scored twice.
    expected = _transition_score([0, 1, 2, 3], [1, 2, 3, 4])

    (actual,) = local_onnx_lm.loglikelihood([request], disable_tqdm=True)

    assert actual[0] == pytest.approx(expected[0], abs=1e-6)
    assert actual[1] == expected[1]


def test_real_onnx_cached_requests_preserve_scores_and_order(local_onnx_lm, tmp_path):
    cached = CachingLM(local_onnx_lm, str(tmp_path / "responses.db"))
    pairs = [("0", " 1 2 3 4"), ("0", " 1 2 3 4"), ("0", " 5")]
    requests = [
        Instance(request_type="loglikelihood", doc={}, arguments=pair, idx=i)
        for i, pair in enumerate(pairs)
    ]
    expected = [
        _transition_score([0, 1, 2, 3], [1, 2, 3, 4]),
        _transition_score([0, 1, 2, 3], [1, 2, 3, 4]),
        _transition_score([0], [5]),
    ]
    try:
        first_results = cached.loglikelihood(requests)
        for actual, reference in zip(first_results, expected, strict=True):
            assert actual[0] == pytest.approx(reference[0], abs=1e-6)
            assert actual[1] == reference[1]
        assert len(cached.dbdict) == 2
        for pair, result in zip(pairs, first_results, strict=True):
            assert cached.dbdict[hash_args("loglikelihood", pair)] == result

        # Cache hits must return the same corrected values without inference.
        def unexpected_inference(tokens):
            pytest.fail("A cached request unexpectedly ran inference")

        local_onnx_lm._forward_logits = unexpected_inference
        assert cached.loglikelihood(list(reversed(requests))) == list(
            reversed(first_results)
        )
    finally:
        cached.dbdict.close()


def test_real_onnx_rolling_loglikelihood(local_onnx_lm):
    tokens = [1, 2, 3, 4, 5, 1, 2, 3, 4]
    request = Instance(
        request_type="loglikelihood_rolling",
        doc={},
        arguments=(" ".join(map(str, tokens)),),
        idx=0,
    )
    expected, _ = _transition_score([0] + tokens[:-1], tokens)

    (actual,) = local_onnx_lm.loglikelihood_rolling([request], disable_tqdm=True)

    assert actual == pytest.approx(expected, abs=2e-6)
