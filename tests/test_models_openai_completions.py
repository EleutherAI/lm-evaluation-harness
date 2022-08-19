import pytest
import os
import json
import openai
import mock
import pickle
import hashlib
import logging

import lm_eval.models as models
from lm_eval.api.utils import set_seed


logger = logging.getLogger(__name__)


def _mock_completion(**kwargs):
    # Mock completion function
    # Loads from a cached+pickled response if it exists, otherwise it will actually try to ping
    os.makedirs("tests/testdata", exist_ok=True)
    arg_hash = hashlib.sha256(
        json.dumps(kwargs, sort_keys=True).encode("utf-8")
    ).hexdigest()
    fname = f"tests/testdata/gpt3_test_{arg_hash}.pkl"
    if os.path.exists(fname):
        with open(fname, "rb") as fh:
            return pickle.load(fh)

    ret = openai.Completion.create(**kwargs)
    ret.api_key = ""
    with open(fname, "wb") as fh:
        pickle.dump(ret, fh)
    return ret


@mock.patch("lm_eval.models.openai_completions.oa_completion", new=_mock_completion)
def test_openai_completions():
    set_seed()
    if "OPENAI_API_SECRET_KEY" not in os.environ:
        os.environ["OPENAI_API_SECRET_KEY"] = ""
    oa_model = models.get_model_from_args_string(
        model_api_name="openai", model_args="engine=ada"
    )
    (
        (ll_dog, ig_dog),
        (ll_cat, ig_cat),
        (_, ll_max_0),
        (_, ll_max_1),
        (_, ll_max_2),
        *vals,
    ) = oa_model.loglikelihood(
        [
            ("The quick brown fox jumps over the lazy", " dog"),
            ("The quick brown fox jumps over the lazy", " cat"),
            ("The quick brown fox jumps over the lazy", ", lazy dog"),
            ("The quick brown fox jumps over the lazy", ", lazy fox"),
            (
                "The quick brown fox jumps over the lazy",
                ", lazy fox and they both fall to the ground",
            ),
            (
                """A mult""",
                """ilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN)""",
            ),
            (
                """The term MLP is used ambiguously, sometimes loosely to any feedforward ANN, sometimes strictly to refer to networks composed of multiple layers of perceptrons""",
                """ (with threshold activation); see § Terminology""",
            ),
            (
                """Multilayer perceptrons are sometimes coll""",
                """oquially referred to as "vanilla" neural networks, especially when they have a single hidden layer.[1]""",
            ),
            (
                """An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear""",
                """ activation function.""",
            ),
            (
                """MLP utilizes a supervised""",
                """ learning technique called backpropagation for training.[2][3] Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.[4]""",
            ),
            (
                """Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic""",
                """ in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. """,
            ),
            (
                """Specifically, we train GPT-3, an autoregressive language model with 175""",
                """ billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.""",
            ),
            (
                """A mult""",
                """ilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN)""",
            ),
            ("""Hello""", """ World"""),
        ]
    )

    assert ll_dog > ll_cat
    assert ig_dog
    assert not ig_cat
    assert not ll_max_0
    assert not ll_max_1
    assert not ll_max_2

    # Test empty context
    oa_model.loglikelihood([("", "test")])
    request_args = {
        "stop_sequences": ["."],
        "max_generation_length": 4,
        "num_fewshot": 0,
    }
    (gen,) = oa_model.greedy_until(
        [("The quick brown fox jumps over the lazy", request_args)]
    )
    assert gen == " dog"

    logger.info([x[0] for x in vals])

    targets = [
        -34.848301606999996,
        -47.148329679999996,
        -45.44380149599999,
        -5.285246016,
        -133.97821690686004,
        -321.2616693239001,
        -658.0299524401041,
        -34.848301606999996,
        -7.525115,
    ]
    for (pred, _), tgt in zip(vals, targets):
        assert pred == pytest.approx(tgt, rel=1e-3)


@mock.patch("lm_eval.models.openai_completions.oa_completion", new=_mock_completion)
def test_openai_completions_perplexity():
    set_seed()
    if "OPENAI_API_SECRET_KEY" not in os.environ:
        os.environ["OPENAI_API_SECRET_KEY"] = ""
    oa_model = models.get_model_from_args_string(
        model_api_name="openai", model_args="engine=ada"
    )
    test_string = "We study empirical scaling laws for language model performance on the cross-entropy loss."
    perplexity = oa_model.loglikelihood_rolling([(test_string,)])[0]
    tgt = -84.38819608
    assert perplexity == pytest.approx(tgt, rel=1e-3)

    # Hack: modify gpt3 to have shorter context length to induce rolling windows
    with mock.patch.object(
        models.openai_completions.OpenAICompletionsLM,
        "max_length",
        new_callable=mock.PropertyMock,
    ) as mock_max_length:
        mock_max_length.return_value = 5
        oa_model = models.get_model_from_args_string(
            model_api_name="openai", model_args="engine=ada"
        )
        perplexity = oa_model.loglikelihood_rolling([(test_string,)])[0]
    tgt = -101.81967209999999
    assert perplexity == pytest.approx(tgt, rel=1e-3)
