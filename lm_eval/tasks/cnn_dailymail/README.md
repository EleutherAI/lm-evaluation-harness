# CNN-DailyMail
## Paper
Teaching Machines to Read and Comprehend https://arxiv.org/abs/1506.03340

The CNN/DailyMail dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. It is widely used for abstractive text summarization tasks. This task uses the non-anonymized Version 3.0.0.

Homepage: https://huggingface.co/datasets/cnn_dailymail

## Citation
```
@inproceedings{hermann2015teaching,
  title={Teaching Machines to Read and Comprehend},
  author={Hermann, Karl Moritz and Kocisky, Tomas and Grefenstette, Edward and Espeholt, Lasse and Kay, Will and Suleyman, Mustafa and Blunsom, Phil},
  booktitle={Advances in Neural Information Processing Systems},
  year={2015}
}
```

## Groups and Tasks
### Groups
* `summarization`

### Tasks
* `cnn_dailymail`: The Version 3.0.0 (non-anonymized) dataset. It evaluates models on their ability to generate multi-sentence abstractive summaries. It uses a zero-shot prompt format (Summarize the following article...) and evaluates using ROUGE (1/2/L) and BERTScore (P/R/F1).

## Checklist
For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?

    * [x] Have you referenced the original paper that introduced the task?

    * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

        * The task uses the standard 3.0.0 (non-anonymized) split from HuggingFace. The implementation uses a standard zero-shot prompt: "Summarize the following article:\n\n{{article}}\n\nSummary:".

        * Evaluation metrics include ROUGE (1, 2, L) and BERTScore (Precision, Recall, F1), matching standard reporting practices for abstractive summarization.

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?

* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?

* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
