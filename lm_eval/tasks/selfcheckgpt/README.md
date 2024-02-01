# Task-name

### Paper

Title: `SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models`

Abstract: `Generative Large Language Models (LLMs) such as GPT-3 are capable of generating highly fluent responses to a wide variety of user prompts. However, LLMs are known to hallucinate facts and make non-factual statements which can undermine trust in their output. Existing fact-checking approaches either require access to the output probability distribution (which may not be available for systems such as ChatGPT) or external databases that are interfaced via separate, often complex, modules. In this work, we propose "SelfCheckGPT", a simple sampling-based approach that can be used to fact-check the responses of black-box models in a zero-resource fashion, i.e. without an external database. SelfCheckGPT leverages the simple idea that if an LLM has knowledge of a given concept, sampled responses are likely to be similar and contain consistent facts. However, for hallucinated facts, stochastically sampled responses are likely to diverge and contradict one another. We investigate this approach by using GPT-3 to generate passages about individuals from the WikiBio dataset, and manually annotate the factuality of the generated passages. We demonstrate that SelfCheckGPT can: i) detect non-factual and factual sentences; and ii) rank passages in terms of factuality. We compare our approach to several baselines and show that our approach has considerably higher AUC-PR scores in sentence-level hallucination detection and higher correlation scores in passage-level factuality assessment compared to grey-box methods.`

`task.py` in this folder uses the original

Homepage: [selfcheckgpt](https://github.com/potsawee/selfcheckgpt)


### Citation

```
@article{manakul2023selfcheckgpt,
  title={Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models},
  author={Manakul, Potsawee and Liusie, Adian and Gales, Mark JF},
  journal={arXiv preprint arXiv:2303.08896},
  year={2023}
}
```

#### Tasks

* `selfcheckgpt`: This task uses generative models to generate wikipedia passage based on given starting topics/words. Then generated passages are messured by [selfcheckgpt](https://github.com/potsawee/selfcheckgpt). The default metric is `SelfCheckNgram`, which does not need GPU. Other metrics are `SelfCheckBERTScore`, `SelfCheckMQAG` and `SelfCheckNLI`, which are model-based scores. You can change the metric by changing the enviornment variables.

The results `"avg-selfcheckgpt` and `max-selfcheckgpt` is the average and max sentences' `selfcheckgpt` score for the generated passage(with temperature=0.0). The score is lower and it is less likely to be hallucination.
```
export SELFCHECKGPTTYPE=SelfCheckBERTScore #SelfCheckMQAG, SelfCheckNLI
```

Since model-based metric are slow when they are running in cpu, you can change the running device to gpu by:
```
export SELFCHECKGPTDEVICE=cuda
```
#### Dependencies for sucessful running
```
pip install spacy
pip install selfcheckgpt
python -m spacy download en
```
### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?








# SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models

In order to run selfcheckgpt evaluation, these dependencies should be installed.
```
pip install spacy
pip install selfcheckgpt
python -m spacy download en
```

selfcheckgpt support different evaluation methods including: `SelfCheckNgram`, `SelfCheckBERTScore`, `SelfCheckMQAG` and `SelfCheckNLI`.
The default evaluation method in llm-eval-harness is `SelfCheckNgram`. You can change the evaluation method by changing the environment variable
```
export SELFCHECKGPTTYPE=SelfCheckNgram
```
For `SelfCheckBERTScore`, `SelfCheckMQAG` and `SelfCheckNLI` evaluation method which will also run some huggingface models, You can change the running device of the selfcheckgpt to GPU by setting enviroment device:
```
export SELFCHECKGPTDEVICE=cuda
```

## Citation

```
@misc{manakul2023selfcheckgpt,
      title={SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models},
      author={Potsawee Manakul and Adian Liusie and Mark J. F. Gales},
      year={2023},
      eprint={2303.08896},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```