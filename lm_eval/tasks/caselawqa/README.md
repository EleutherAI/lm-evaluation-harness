# Task-name

CaselawQA

### Paper

Title: Lawma: The Power of Specialization for Legal Tasks

Abstract: https://arxiv.org/abs/2407.16615

CaselawQA is a benchmark comprising legal classification tasks, drawing from the Supreme Court and Songer Court of Appeals legal databases.
The majority of its 10,000 questions are multiple-choice, with 5,000 sourced from each database. 
The questions are randomly selected from the test sets of the [Lawma tasks](https://huggingface.co/datasets/ricdomolm/lawma-tasks).\
From a technical machine learning perspective, these tasks provide highly non-trivial classification problems where even the best models leave much room for improvement. 
From a substantive legal perspective, efficient solutions to such classification problems have rich and important applications in legal research.
Homepage: https://github.com/socialfoundations/lawma

**Impotant**: it is strongly recommended to use --apply_chat_template, e.g.

```bash
lm_eval --model vllm --model_args pretrained=ricdomolm/lawma-8b,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.85,data_parallel_size=1 --tasks caselawqa --batch_size auto --apply_chat_template
``` 


### Citation

```
@misc{dominguezolmedo2024lawmapowerspecializationlegal,
      title={Lawma: The Power of Specialization for Legal Tasks}, 
      author={Ricardo Dominguez-Olmedo and Vedant Nanda and Rediet Abebe and Stefan Bechtold and Christoph Engel and Jens Frankenreiter and Krishna Gummadi and Moritz Hardt and Michael Livermore},
      year={2024},
      eprint={2407.16615},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.16615}, 
}
```

### Groups, Tags, and Tasks

#### Groups

* `caselawqa`: average accuracy of `caselawqa_sc` and `caselaw_songer`
* `caselawqa_cot`: average accuracy of `caselawqa_sc_cot` and `caselaw_songer_cot`

#### Tags

#### Tasks

* `caselawqa_sc`: 5,000 questions derived from the Supreme Court database
* `caselaw_songer`: 5,000 questions derived from the Songer Court of Appeals database
* `caselawqa_tiny`: 1145 questions subsampled from `caselawqa_sc` and `caselawqa_songer`
* `caselawqa_sc_cot`: same as `caselawqa_sc` but with a Chain of Thought prompt
* `caselawqa_songer_cot`: same as `caselawqa_songer` but with a Chain of Thought prompt
* `caselawqa_tiny_cot`: same as `caselawqa_tiny` but with a Chain of Thought prompt

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
