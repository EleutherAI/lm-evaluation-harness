# TalkFa

## Paper

Title: `TalkFa: A Unified Benchmark for Farsi Dialogue Generation and Understanding`

Paper: https://arxiv.org/abs/2609.01810

TalkFa is a Farsi dialogue benchmark with three human-reviewed datasets: Wikipedia-grounded
six-turn dialogues (WikiFaDial), utterances labeled for dialogue acts and emotions
(DailyDialog-Fa), and theatrical dialogues labeled for sentiment (PlayDial-Fa).

Homepage: https://github.com/KamyarZeinalipour/TalkFa

Data: https://huggingface.co/datasets/Kamyar-zeinalipour/TalkFa

### Citation

```text
@misc{jamshidi2026talkfaunifiedbenchmarkfarsi,
      title={TalkFa: A Unified Benchmark for Farsi Dialogue Generation and Understanding},
      author={Neda Jamshidi and Kamyar Zeinalipour and Fahimeh Akbari and Monica Bianchini and Marco Maggini and Marco Gori},
      year={2026},
      eprint={2609.01810},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.01810},
}
```

### Groups, Tags, and Tasks

#### Groups

* `talkfa`: the three understanding tasks, length-normalized accuracy averaged over tasks.

#### Tasks

* `talkfa_act`: dialogue-act classification of an utterance (inform, question, directive, commissive); accuracy, length-normalized accuracy and macro F1.
* `talkfa_emotion`: emotion classification of an utterance (seven classes, mostly "no emotion"); accuracy, length-normalized accuracy and macro F1.
* `talkfa_sentiment`: sentiment of a theatrical dialogue (negative, neutral, positive); accuracy, length-normalized accuracy and macro F1.
* `talkfa_dialogue`: turn a Wikipedia paragraph into a six-turn informal dialogue; chrF against the reference dialogue.

Class labels are scored as Farsi words. Their lengths differ (for example the two-word "no emotion" label), so `acc_norm`, which normalizes the log-likelihood by length, is the recommended accuracy.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

These tasks score labels by log-likelihood and dialogues by chrF. The paper's own evaluation
scripts are in the GitHub repository; numbers from the two setups are not directly comparable.

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* v1.0: initial version.
