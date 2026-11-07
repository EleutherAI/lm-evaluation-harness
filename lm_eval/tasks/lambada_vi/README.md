# LAMBADA (Vietnamese)

### Paper
The LAMBADA dataset: Word prediction requiring a broad discourse context
https://arxiv.org/pdf/1606.06031.pdf

ViLLM-Eval: A Comprehensive Evaluation Suite for Vietnamese Large Language Models
https://arxiv.org/abs/2404.11086

LAMBADA is a dataset to evaluate the capabilities of computational models for text
understanding by means of a word prediction task. LAMBADA is a collection of narrative
passages sharing the characteristic that human subjects are able to guess their last
word if they are exposed to the whole passage, but not if they only see the last
sentence preceding the target word. To succeed on LAMBADA, computational models
cannot simply rely on local context, but must be able to keep track of information
in the broader discourse.

LAMBADA_vi follows the same methodology for Vietnamese: passages of 4-6 sentences
sourced from Vietnamese Wikipedia, where the model must predict the final word of
the passage. It was introduced as part of ViLLM-Eval, the evaluation suite of the
Vietnamese Large Language Models challenge at VLSP 2023. Unlike the
machine-translated LAMBADA variants, the passages here are natively Vietnamese, so
the task uses the dataset's own `context` (passage without the final word) and
`target_word` fields instead of splitting the text on spaces.

Homepage: https://huggingface.co/datasets/vlsp-2023-vllm/lambada_vi

### Groups and Tasks

#### Groups

* (none — evaluate the task directly by name, `lambada_vi`)

#### Tasks

* `lambada_vi`: Vietnamese LAMBADA-style word prediction (10,000 test passages
  from Vietnamese Wikipedia), as used in ViLLM-Eval (VLSP 2023).

### Citation

@misc{
    author={Paperno, Denis and Kruszewski, Germán and Lazaridou, Angeliki and Pham, Quan Ngoc and Bernardi, Raffaella and Pezzelle, Sandro and Baroni, Marco and Boleda, Gemma and Fernández, Raquel},
    title={The LAMBADA dataset},
    DOI={10.5281/zenodo.2630551},
    publisher={Zenodo},
    year={2016},
    month={Aug}
}

@misc{nguyen2024villmeval,
    title={ViLLM-Eval: A Comprehensive Evaluation Suite for Vietnamese Large Language Models},
    author={Trong-Hieu Nguyen and Anh-Cuong Le and Viet-Cuong Nguyen},
    year={2024},
    eprint={2404.11086},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
(This task follows the reference implementation in the VLSP 2023 fork of the
harness, which defined the same task with `context` as the prompt and
" " + target_word` as the target.)


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
