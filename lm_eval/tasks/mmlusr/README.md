# MMLU-SR

## Paper
Title: [Reasoning or Simply Next Token Prediction? A Benchmark for Stress-Testing Large Language Models](https://arxiv.org/abs/2406.15468v1)


We propose MMLU-SR, a novel dataset designed to measure the true comprehension abilities of Large Language Models (LLMs) by challenging their performance in question-answering tasks with modified terms. We reasoned that an agent that ``truly'' understands a concept can still evaluate it when key terms are replaced by suitably defined alternate terms, and sought to differentiate such comprehension from mere text replacement. In our study, we modified standardized test questions by replacing a key term with a dummy word along with its definition. The key term could be in the context of questions, answers, or both questions and answers.
Notwithstanding the high scores achieved by recent popular LLMs on the MMLU leaderboard, we found a substantial reduction in model performance after such replacement, suggesting poor comprehension. This new benchmark provides a rigorous benchmark for testing true model comprehension, and poses a challenge to the broader scientific community.

Github Homepage: [https://github.com/Wang-ML-Lab/MMLU-SR](https://github.com/Wang-ML-Lab/MMLU-SR)
Huggingface Dataset: [https://huggingface.co/datasets/NiniCat/MMLU-SR]([https://huggingface.co/datasets/NiniCat/MMLU-SR)


## Citation
```bib
@misc{wang2024reasoningsimplytokenprediction,
      title={Reasoning or Simply Next Token Prediction? A Benchmark for Stress-Testing Large Language Models},
      author={Wentian Wang and Paul Kantor and Jacob Feldman and Lazaros Gallos and Hao Wang},
      year={2024},
      eprint={2406.15468},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.15468},
}
```

### Groups and Tasks

#### Groups

- `mmlusr`: MMLU variant where the terminology in the question and answers are modified.
- `mmlusr_answer_only`: MMLU variant where the terminology in the answers are modified.
- `mmlusr_question_only`: MMLU variant where the terminology in the question is modified.

#### Tasks

There are 57 symbol replaced subjects in each group. You can run a single task by:

* `mmlusr_question_only_abstract_algebra`

Or by categories:

* `mmlusr_question_only_stem_tasks `


### Checklist

The checklist is the following:

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    * The implementation in the original paper is one where the model is first fine-tuned on the data. They do have a few-shot evaluation for GPT-3, however the few-shot context used here is sourced from [Lewkowycz et al](https://arxiv.org/abs/2206.14858). The achieved accuracy on Llama-2 models is comparable to that provided in the paper, though not identical.


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

### Variant Wishlist

- [ ] zero-shot variant
