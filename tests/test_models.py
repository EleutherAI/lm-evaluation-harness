import hashlib
import json
import openai
import os
import pickle
import pytest
import unittest.mock as mock

import lm_eval.models as models


LOGLIKELIHOOD_TEST_CASES = [
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


# Test HuggingFace Models (GPT-2)


def test_gpt2():
    gpt2 = models.get_model("gpt2").create_from_arg_string("device=cpu")
    (
        (ll_dog, ig_dog),
        (ll_cat, ig_cat),
        (_, ll_max_0),
        (_, ll_max_1),
        (_, ll_max_2),
        *vals,
    ) = gpt2.loglikelihood(LOGLIKELIHOOD_TEST_CASES)

    assert ll_dog > ll_cat
    assert not ig_cat

    assert not ll_max_0
    assert ll_max_1
    assert ll_max_2

    # test empty context
    gpt2.loglikelihood([("", "test")])

    (gen,) = gpt2.greedy_until(
        [("The quick brown fox jumps over the lazy", [".", "\n"])]
    )

    assert gen == ", lazy fox and they both fall to the ground"

    targets = [
        -61.60536193847656,
        -56.57843780517578,
        -62.131004333496094,
        -9.799489974975586,
        -153.96334838867188,
        -341.222900390625,
        -731.1475830078125,
        -61.60536193847656,
        -8.682319641113281,
    ]

    for (pred, _), tgt in zip(vals, targets):
        assert pred == pytest.approx(tgt, rel=1e-3)


def test_gpt2_perplexity():
    gpt2 = models.get_model("gpt2").create_from_arg_string("device=cpu")
    test_string = "We study empirical scaling laws for language model performance on the cross-entropy loss."
    perplexity = gpt2.loglikelihood_rolling([(test_string,)])[0]
    tgt = sum(
        [
            -4.9599953,
            -8.069298,
            -8.308624,
            -10.178513,
            -8.906924,
            -1.9318912,
            -7.745445,
            -7.146077,
            -5.2072,
            -3.5882986,
            -1.9957212,
            -8.044922,
            -0.20841774,
            -5.1096807,
            -0.099879116,
            -8.888423,
            -4.6180487,
        ]
    )
    assert perplexity == pytest.approx(tgt, rel=1e-3)

    with mock.patch.object(
        models.gpt2.HFLM, "max_length", new_callable=mock.PropertyMock
    ) as mock_max_length:
        mock_max_length.return_value = 5
        gpt2 = models.get_model("gpt2").create_from_arg_string("device=cpu")
        perplexity = gpt2.loglikelihood_rolling([(test_string,)])[0]
    tgt = sum(
        [
            -4.96001,
            -8.069275,
            -8.308612,
            -10.178482,
            -8.90691,
            -4.037338,
            -8.09261,
            -11.662385,
            -10.206891,
            -4.425003,
            -2.2563353,
            -7.909143,
            -1.9304147,
            -7.3610134,
            -2.3120654,
            -7.3229,
            -2.1643813,
        ]
    )
    assert perplexity == pytest.approx(tgt, rel=1e-3)


# Test OpenAI Models (GPT-3)


def openai_mock_completion(**kwargs):
    # Mock completion function
    # Loads from a cached+pickled response if it exists, otherwise it will actually try to ping
    os.makedirs("tests/testdata", exist_ok=True)
    hash = hashlib.sha256(
        json.dumps(kwargs, sort_keys=True).encode("utf-8")
    ).hexdigest()
    fname = f"tests/testdata/gpt3_test_{hash}.pkl"

    if os.path.exists(fname):
        with open(fname, "rb") as fh:
            return pickle.load(fh)
    ret = openai.Completion.create(**kwargs)
    ret.api_key = ""
    with open(fname, "wb") as fh:
        pickle.dump(ret, fh)
    return ret


@mock.patch("lm_eval.models.gpt3.oa_completion", new=openai_mock_completion)
def test_gpt3():
    if "OPENAI_API_SECRET_KEY" not in os.environ:
        os.environ["OPENAI_API_SECRET_KEY"] = ""
    gpt3 = models.get_model("gpt3").create_from_arg_string("engine=ada")
    (
        (ll_dog, ig_dog),
        (ll_cat, ig_cat),
        (_, ll_max_0),
        (_, ll_max_1),
        (_, ll_max_2),
        *vals,
    ) = gpt3.loglikelihood(LOGLIKELIHOOD_TEST_CASES)

    assert ll_dog > ll_cat
    assert not ig_cat

    assert ig_dog
    assert not ll_max_0
    assert not ll_max_1
    assert not ll_max_2

    # test empty context
    gpt3.loglikelihood([("", "test")])

    (gen,) = gpt3.greedy_until(
        [("The quick brown fox jumps over the lazy", [".", "\n"])]
    )

    assert gen == " dog"

    print([x[0] for x in vals])

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


@mock.patch("lm_eval.models.gpt3.oa_completion", new=openai_mock_completion)
def test_gpt3_perplexity():
    if "OPENAI_API_SECRET_KEY" not in os.environ:
        os.environ["OPENAI_API_SECRET_KEY"] = ""
    gpt3 = models.get_model("gpt3").create_from_arg_string("engine=ada")
    test_string = "We study empirical scaling laws for language model performance on the cross-entropy loss."
    perplexity = gpt3.loglikelihood_rolling([(test_string,)])[0]
    tgt = -84.38819608
    assert perplexity == pytest.approx(tgt, rel=1e-3)

    # Hack: modify gpt3 to have shorter context length to induce rolling windows
    with mock.patch.object(
        models.gpt3.GPT3LM, "max_length", new_callable=mock.PropertyMock
    ) as mock_max_length:
        mock_max_length.return_value = 5
        gpt3 = models.get_model("gpt3").create_from_arg_string("engine=ada")
        perplexity = gpt3.loglikelihood_rolling([(test_string,)])[0]
    tgt = -101.81967209999999
    assert perplexity == pytest.approx(tgt, rel=1e-3)


# Test TextSynth Models (GPT-J)


def textsynth_mock_completion(**kwargs):
    # Mock completion function
    # Loads from a cached+pickled response if it exists, otherwise it will actually try to ping
    import requests

    os.makedirs("tests/testdata", exist_ok=True)
    hash_kwargs = {k: v for k, v in kwargs.items() if k != "headers"}
    hash = hashlib.sha256(
        json.dumps(hash_kwargs, sort_keys=True).encode("utf-8")
    ).hexdigest()
    fname = f"tests/testdata/textsynth_test_{hash}.pkl"

    if os.path.exists(fname):
        with open(fname, "rb") as fh:
            return pickle.load(fh)
    ret = requests.post(**kwargs)
    with open(fname, "wb") as fh:
        pickle.dump(ret, fh)
    return ret


@mock.patch(
    "lm_eval.models.textsynth.textsynth_completion", new=textsynth_mock_completion
)
def test_textsynth():
    if "TEXTSYNTH_API_SECRET_KEY" not in os.environ:
        os.environ["TEXTSYNTH_API_SECRET_KEY"] = ""
    textsynth = models.get_model("textsynth").create_from_arg_string("engine=gptj_6B")
    (
        (ll_dog, ig_dog),
        (ll_cat, ig_cat),
        (_, ll_max_0),
        (_, ll_max_1),
        (_, ll_max_2),
        *vals,
    ) = textsynth.loglikelihood(LOGLIKELIHOOD_TEST_CASES)

    assert ll_dog > ll_cat
    assert not ig_cat

    assert ig_dog
    assert not ll_max_0
    assert not ll_max_1
    assert not ll_max_2

    # test empty context
    textsynth.loglikelihood([("", "test")])

    (gen,) = textsynth.greedy_until(
        [("The quick brown fox jumps over the lazy", [".", "\n"])]
    )

    assert gen == " dog"

    print([x[0] for x in vals])

    targets = [
        -17.90513712817,
        -41.83518912287,
        -33.82445643841,
        -2.377361565302,
        -99.53018069754,
        -243.5642283598,
        -528.6862613790,
        -17.90513712817,
        -5.041000672142,
    ]

    for (pred, _), tgt in zip(vals, targets):
        assert pred == pytest.approx(tgt, rel=1e-3)
