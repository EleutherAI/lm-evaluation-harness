# Humaneval-Infilling

### Paper

Title: Efficient Training of Language Models to Fill in the Middle
Abstract: https://arxiv.org/pdf/2207.14255

We show that autoregressive language models can learn to infill text after we apply a straightforward transformation to the dataset, which simply moves a span of text from the middle of a document to its end. While this data augmentation has garnered much interest in recent years, we provide extensive evidence that training models with a large fraction of data transformed in this way does not harm the original left-to-right generative capability, as measured by perplexity and sampling evaluations across a wide range of scales. Given the usefulness, simplicity, and efficiency of training models to fill-in-the-middle (FIM), we suggest that future autoregressive language models be trained with FIM by default. To this end, we run a series of ablations on key hyperparameters, such as the data transformation frequency, the structure of the transformation, and the method of selecting the infill span. We use these ablations to prescribe strong default settings and best practices to train FIM models. We have released our best infilling model trained with best practices in our API, and release our infilling benchmarks to aid future research.

Homepage: https://github.com/openai/human-eval-infilling


### Citation

```
@article{bavarian2022efficient,
  title={Efficient Training of Language Models to Fill in the Middle},
  author={Bavarian, Mohammad and Jun, Heewoo and Tezak, Nikolas and Schulman, John and McLeavey, Christine and Tworek, Jerry and Chen, Mark},
  journal={arXiv preprint arXiv:2207.14255},
  year={2022}
}
```

### Groups and Tasks

#### Groups

- `humaneval_infilling`

This dataset has 4 subsets: HumanEval-MultiLineInfilling, HumanEval-SingleLineInfilling, HumanEval-RandomSpanInfilling, HumanEval-RandomSpanInfillingLight. The single-line, multi-line, random span infilling and its light version have 1033, 5815, 1640 and 164 tasks, respectively.

#### Tasks

- `humaneval_single_line_infilling`
- `humaneval_multi_line_infilling`
- `humaneval_random_span_infilling`
- `humaneval_random_span_infilling_light`

### Checklist

For adding novel benchmarks/datasets to the library:

- [ ] Is the task an existing benchmark in the literature?
  - [ ] Have you referenced the original paper that introduced the task?
  - [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

- [ ] Is the "Main" variant of this task clearly denoted?
- [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
- [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
