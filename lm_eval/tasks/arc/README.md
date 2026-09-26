# ARC

### Paper

Title: Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge

Abstract: https://arxiv.org/abs/1803.05457

The ARC dataset consists of 7,787 science exam questions drawn from a variety
of sources, including science questions provided under license by a research
partner affiliated with AI2. These are text-only, English language exam questions
that span several grade levels as indicated in the files. Each question has a
multiple choice structure (typically 4 answer options). The questions are sorted
into a Challenge Set of 2,590 “hard” questions (those that both a retrieval and
a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.

Homepage: https://allenai.org/data/arc


### Citation

ARC:

```
@article{Clark2018ThinkYH,
  title={Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
  author={Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
  journal={ArXiv},
  year={2018},
  volume={abs/1803.05457}
}
```

Set-LLM (motivation for the permutation-robustness setup):

```
@inproceedings{Egressy2025SetLLM,
  title={Set-LLM: A Permutation-Invariant LLM},
  author={Beni Egressy and Jan St{\"u}hmer},
  booktitle={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025},
  url={https://arxiv.org/abs/2505.15433}
}
```

### Groups, Tags, and Tasks

#### Groups

None.

#### Tags

* `ai2_arc`: Evaluates `arc_easy` and `arc_challenge`

#### Tasks

* `arc_easy`
* `arc_challenge`
* `arc_challenge_choices_permuted`

`arc_challenge_choices_permuted` is a zero-shot variant that shows every answer
choice without a letter label and evaluates every ordering of the choices. The
model scores the text of each answer choice after the full question and choice
list. `acc` averages accuracy over permutations within each question, then over
questions; `acc_adv` counts a question as correct only if every permutation is
correct. `acc_norm` and `acc_norm_adv` use character-length-normalized answer
log likelihoods, as in the standard harness multiple-choice metric. This task
is motivated by [Set-LLM](https://arxiv.org/abs/2505.15433), but does not
implement its position encoding or attention mask.

### Checklist

For adding novel benchmarks/datasets to the library:

Not applicable: this variant uses the existing ARC dataset rather than adding
a new benchmark or dataset.

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted? (`arc_challenge`)
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant? (None exactly; motivated by Set-LLM.)
