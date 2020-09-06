# Evaluation Harness for Large Language Models

## Overview 

The goal of this project is to build a set of tools for evaluating LMs on typical NLU tasks, based on evaluation of GPT-3 as described in https://arxiv.org/pdf/2005.14165.pdf. Following the initial description, this repo should support 3 functions:
1. LM Evaluation
2. Removing task val/test data from LM training set
3. Adding task training data to LM training set

The raw Google doc can be found here: https://docs.google.com/document/d/177dwJpH8GHebISXYZSn4NL98sXdCtQMH82b7O5F7jmw/edit?usp=sharing

### 1. LM Evaluation
Given an LM, we want to evaluate it on a wide range of NLU tasks. We should at least cover the set of tasks in the GPT-3 paper, and any other tasks/benchmarks that are relevant. We will follow the GPT-3 format of a) zero-shot, b) one-shot, c) few-shot evaluation.

To do this, we need 3 components:
* Data downloader (shared with later sections, potentially needs to be directly linked to the latter 2 components)
* Task formatter
* Task evaluator

The **data downloader** should download data for the relevant tasks.
* We should heavily rely on Hugging Face's NLP for this. They are already doing most of the work with handling data scripts/caching.
* Optionally, we can rely directly on HF-NLP's caching, but that makes it awkward to handle non-HF-NLP datasets. Otherwise, we can just write them out to .jsonl. My feeling is that NLU data storage will be a drop in the bucket compared to LM data.
* Where we're not using HF-NLP, we can keep the data in the raw format (.jsonl, tsv, etc) and let the other components handle transforming it.

The **task formatter** formats the task input data into an LM-usable format.
* We should potentially support multiple formats for a given task, e.g. some formats may be better or worse suited for LM evaluation. See also: prompt-engineering
* The task formatter should also support zero/one/few-shot packing of training examples into an input. This may require weird interactions with the tokenizer for dealing with max-token issues.

The **task evaluator** scores a task.
* In essence, we want to generation output predictions for all our input examples, and feed them into some function that pops out a score (or scores)
An alternative approach is to collect the output logits and score them against the expected set of outputs.
* Some tasks have weird evaluation schemes, so we should make this as general as possible.
* Will thus likely have to be closely tied with the formatter.
* Likewise, we should take advantage of HF-NLP's metrics.
We might as well provide a sufficiently general API for the model to support OpenAI API as well. This can double up as an effort to reproduce the OpenAI NLU results.

### 2. Removing val/test data from LM training set
With the data downloader in place, we simply need to (1) expose the val/test examples, and (2) remove them from the training set.

* Arguably, (2) should be handled by LM preprocessing in a more general way. There are probably non-NLU-eval cases where we want to remove some specific data from training.
* Depending on how exactly we do the val/test removal, we may want to format the same example multiple ways to ensure that they don't get leaked into the training set in a slightly tweaked format.
* Thought experiment: SQuAD is based largely on Wikipedia. What exactly would we want to remove from the LM?
* [GPT-3]: In GPT-3, they attempted to remove val/test from their LM set, but there was a bug that caused leakage. So they ended up doing the opposite: removing overlaps from the LM set from the val/test. Funky.
* [GPT-3]: See page 30 and Appendix C for details. They do some funky n-gram based search and removal. We should think about whether we want to follow their protocol exactly

### 3. Adding task training data to LM training set
This part is the easiest. I guess we just write out some text files containing the training data? We can let the usual LM preprocessing pipeline handle it from there.
