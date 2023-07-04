# New Model Guide

The `lm-evaluation-harness` is intended to be a model-agnostic framework for evaluating . We provide first-class support for HuggingFace `AutoModelForCausalLM` and `AutoModelForSeq2SeqLM` type models, but

This guide may be of special interest to users who are using the library outside of the repository, via installing the library via pypi and calling `lm_eval.evaluator.evaluate()` to evaluate an existing model.

In order to properly evaluate a given LM, we require implementation of a wrapper class subclassing the `lm_eval.api.model.LM` class, that defines how the Evaluation Harness should interface with your model. This guide walks through how to write this `LM` subclass via adding it to the library!

## Setup

To get started contributing, go ahead and fork the main repo, clone it, create a branch with the name of your task, and install the project requirements in your environment:

```sh
# After forking...
git clone https://github.com/<YOUR-USERNAME>/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout big-refactor
git checkout -b <model-type>
pip install -e ".[dev]"
```

Now, we'll create a new file where we'll be adding our model:

```sh
touch lm_eval/models/<my_model_filename>.py
```

**Tip: this filename should not shadow package names! For example, naming your file `anthropic.py` is disallowed since the API's name on pypi is `anthropic`, but naming it `anthropic_llms.py` works with no problems.**

## Interface

All models must subclass the `lm_eval.api.model.LM` class.

The LM class enforces a common interface via which we can extract responses from a model:

```python
class MyCustomLM(LM):
    #...
    def loglikelihood(self, requests):


    def loglikelihood_rolling(self, requests):


    def greedy_until(self, requests):
    #...
```

We support

The three types of



smth smth tokenizer-agnostic

3 reqtypes
- greedy_until, and the arguments passed to it

- loglikelihood, and args passed to it

- loglikelihood_rolling, and args passed to it


## Registration

Congrats on implementing your model! Now it's time to test it out.

To make your model usable via the command line interface to `lm-eval` using `main.py`, you'll need to tell `lm-eval` what your model's name is.

This is done via a *decorator*, `lm_eval.api.registry.register_model`. Using `register_model()`, one can both tell the package what the model's name(s) to be used are when invoking it with `python main.py --model <name>` and alert `lm-eval` to the model's existence.

```python
from lm_eval.api.registry import register_model

@register_model("<name1>", "<name2>")
class MyCustomLM(LM):
```

Using this decorator results in the class being added to an accounting of the usable LM types maintained internally to the library at `lm_eval.api.registry.MODEL_REGISTRY`. See `lm_eval.api.registry` for more detail on what sorts of registries and decorators exist in the library!



## Other

**Pro tip**: In order to make the Evaluation Harness overestimate total runtimes rather than underestimate it, HuggingFace models come in-built with the ability to provide responses on data points in *descending order by total input length* via `lm_eval.utils.Reorderer`. Take a look at `lm_eval.models.hf_causal.HFLM` to see how this is done, and see if you can implement it in your own model!

## Conclusion

After reading this guide, you should be able to add new model APIs or implementations to the Eval Harness library!
