# mmlu_pro

### Paper

Title: `MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark`

Abstract: `In the age of large-scale language models, benchmarks like the Massive Multitask Language Understanding (MMLU) have been pivotal in pushing the boundaries of what AI can achieve in language comprehension and reasoning across diverse domains. However, as models continue to improve, their performance on these benchmarks has begun to plateau, making it increasingly difficult to discern differences in model capabilities. This paper introduces MMLU-Pro, an enhanced dataset designed to extend the mostly knowledge-driven MMLU benchmark by integrating more challenging, reasoning-focused questions and expanding the choice set from four to ten options. Additionally, MMLU-Pro eliminates the trivial and noisy questions in MMLU. Our experimental results show that MMLU-Pro not only raises the challenge, causing a significant drop in accuracy by 16% to 33% compared to MMLU but also demonstrates greater stability under varying prompts. With 24 different prompt styles tested, the sensitivity of model scores to prompt variations decreased from 4-5% in MMLU to just 2% in MMLU-Pro. Additionally, we found that models utilizing Chain of Thought (CoT) reasoning achieved better performance on MMLU-Pro compared to direct answering, which is in stark contrast to the findings on the original MMLU, indicating that MMLU-Pro includes more complex reasoning questions. Our assessments confirm that MMLU-Pro is a more discriminative benchmark to better track progress in the field.`

Note: the original mmlu_pro dataset stores categories as a column instead of subsets. Furthermore, the original dataset is split into `test` and `validation (dev)`. To render it compatible with existing lm-evaluation-harness codebase, the following preprocessing steps are taken:
- create subsets for different categories
- shuffle and split existing `test` split into `test` and `validation` splits with a 90:10 ratio, similar to mmlu's approach

Homepage (original): https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
Homepage (preprocessed): https://huggingface.co/datasets/sjyuxyz/MMLU-Pro-with-subset

### Citation

```bibtex
@misc{wang2024mmlupro,
      title={MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark}, 
      author={Yubo Wang and Xueguang Ma and Ge Zhang and Yuansheng Ni and Abhranil Chandra and Shiguang Guo and Weiming Ren and Aaran Arulraj and Xuan He and Ziyan Jiang and Tianle Li and Max Ku and Kai Wang and Alex Zhuang and Rongqi Fan and Xiang Yue and Wenhu Chen},
      year={2024},
      eprint={2406.01574},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```

### Groups and Tasks

#### Groups

* `mmlu_pro`: 'All 14 subjects of the mmlu_pro dataset, evaluated following the methodology in mmlu's original implementation'
* `mmlu_pro_flan_cot_fewshot`: 'mmlu_pro_flan_cot_fewshot includes 5-shot of exemplars for chain-of-thought approach'
* `mmlu_pro_flan_cot_zeroshot`: 'mmlu_pro_flan_cot_zeroshot evaluates using zero-shot chain-of-thought approach'
* `mmlu_pro_generative`:  'mmlu_pro_generative solves questions of mmlu_pro using direct (generative) approach'
* `mmlu_pro_continuation`: 'mmlu_pro_continuation evaluates the ability to continue and complete a given text'

#### Tasks

The following tasks evaluate subjects in the mmlu_pro dataset
- `mmlu_pro_{subject_english}`
- `mmlu_pro_flan_cot_fewshot_{subject_english}`
- `mmlu_pro_flan_cot_zeroshot_{subject_english}`
- `mmlu_pro_generative_{subject_english}`
- `mmlu_pro_continuation_{subject_english}`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
