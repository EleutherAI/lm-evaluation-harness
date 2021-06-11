import lm_eval.tasks as tasks
import lm_eval.models as models
import lm_eval.evaluator as evaluator
import random
import pytest
import os
import json
import openai
import mock
import pickle
import hashlib

os.environ['OPENAI_API_SECRET_KEY'] = ""


def completion(**kwargs):
    hash = hashlib.sha256(json.dumps(kwargs, sort_keys=True).encode('utf-8')).hexdigest()
    fname = f"tests/testdata/gpt3_test_{hash}.pkl"

    if os.path.exists(fname):
        with open(fname, 'rb') as fh:
            return pickle.load(fh)
    ret = openai.Completion.create(**kwargs)
    with open(fname, 'wb') as fh:
        pickle.dump(ret, fh)
    return ret


os.makedirs("tests/testdata", exist_ok=True)


@mock.patch("lm_eval.models.gpt3.oa_completion", new=completion)
def test_gpt3():
    gpt3 = models.get_model('gpt3').create_from_arg_string("engine=ada")
    (ll_dog, ig_dog), (ll_cat, ig_cat), (_, ll_max_0), (_, ll_max_1), (_, ll_max_2), *vals = gpt3.loglikelihood([
        ('The quick brown fox jumps over the lazy', ' dog'),
        ('The quick brown fox jumps over the lazy', ' cat'),
        ('The quick brown fox jumps over the lazy', ', lazy dog'),
        ('The quick brown fox jumps over the lazy', ', lazy fox'),
        ('The quick brown fox jumps over the lazy', ', lazy fox and they both fall to the ground'),
        
        ("""A mult""", """ilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN)"""), 
        ("""The term MLP is used ambiguously, sometimes loosely to any feedforward ANN, sometimes strictly to refer to networks composed of multiple layers of perceptrons""", """ (with threshold activation); see § Terminology"""), 
        ("""Multilayer perceptrons are sometimes coll""", """oquially referred to as "vanilla" neural networks, especially when they have a single hidden layer.[1]"""), 
        ("""An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear""", """ activation function."""), 
        ("""MLP utilizes a supervised""", """ learning technique called backpropagation for training.[2][3] Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.[4]"""), 
        ("""Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic""", """ in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. """), 
        ("""Specifically, we train GPT-3, an autoregressive language model with 175""", """ billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general."""), 
        ("""A mult""", """ilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN)"""), 
        ("""Hello""", """ World"""), 
    ])

    assert ll_dog > ll_cat
    assert not ig_cat

    assert ig_dog
    assert not ll_max_0
    assert not ll_max_1
    assert not ll_max_2

    # test empty context
    gpt3.loglikelihood([('', 'test')])

    gen, = gpt3.greedy_until([
        ('The quick brown fox jumps over the lazy', ['.', '\n'])
    ])

    assert gen == ' dog'

    print([x[0] for x in vals])

    targets = [-34.85833048, -47.114367866, -45.43520782100001, -5.289627985, -133.96879783896998, -321.30299892039994, -658.0542459504098, -34.85833048, -7.5162964]

    for (pred, _), tgt in zip(vals, targets):
        assert pred == pytest.approx(tgt, rel=1e-3)



@mock.patch("lm_eval.models.gpt3.oa_completion", new=completion)
def test_gpt3_perplexity():
    gpt3 = models.get_model('gpt3').create_from_arg_string("engine=ada")
    test_string = "We study empirical scaling laws for language model performance on the cross-entropy loss."
    perplexity = gpt3.loglikelihood_rolling([(test_string,)])[0]
    tgt = -84.38819608
    assert perplexity == pytest.approx(tgt, rel=1e-3)

    # Hack: modify gpt3 to have shorter context length to induce rolling windows
    gpt3.MAX_LENGTH = 5
    perplexity = gpt3.loglikelihood_rolling([(test_string,)])[0]
    tgt = -101.93490880000002
    assert perplexity == pytest.approx(tgt, rel=1e-3)
