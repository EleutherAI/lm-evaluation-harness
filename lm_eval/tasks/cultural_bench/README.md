# CulturalBench

## Paper

Title: CulturalBench: a Robust, Diverse and Challenging Benchmark on Measuring the (Lack of) Cultural Knowledge of LLMs

Abstract: https://arxiv.org/abs/2410.02677

Homepage: https://huggingface.co/datasets/kellycyy/CulturalBench

CulturalBench is a set of 1,227 human-written and human-verified questions for effectively assessing LLMs’ cultural knowledge, covering 45 global regions including the underrepresented ones like Bangladesh, Zimbabwe, and Peru.

We evaluate models on two setups: CulturalBench-Easy and CulturalBench-Hard which share the same questions but asked differently.

- CulturalBench-Easy: multiple-choice questions (Output: one out of four options i.e. A,B,C,D). Evaluate model accuracy at question level (i.e. per question_idx). There are 1,227 questions in total.
- CulturalBench-Hard: binary (Output: one out of two possibilties i.e. True/False). Evaluate model accuracy at question level (i.e. per question_idx). There are 1,227x4=4908 binary judgements in total with 1,227 questions provided.


### Citation

```text
@misc{chiu2024culturalbenchrobustdiversechallenging,
      title={CulturalBench: a Robust, Diverse and Challenging Benchmark on Measuring the (Lack of) Cultural Knowledge of LLMs}, 
      author={Yu Ying Chiu and Liwei Jiang and Bill Yuchen Lin and Chan Young Park and Shuyue Stella Li and Sahithya Ravi and Mehar Bhatia and Maria Antoniak and Yulia Tsvetkov and Vered Shwartz and Yejin Choi},
      year={2024},
      eprint={2410.02677},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.02677}, 
}```

### Groups, Tags, and Tasks

#### Groups

- `cultural_bench`: Overall group combining both easy and hard variants
- `cultural_bench_easy`: Group of all easy (A,B,C,D) country tasks  
- `cultural_bench_hard`: Group of all hard (True/False) country tasks

#### Tasks

Countries: Argentina, Australia, Bangladesh, Brazil, Canada, Chile, China, Czech Republic, Egypt, France, Germany, Hong Kong, India, Indonesia, Iran, Israel, Italy, Japan, Lebanon, Malaysia, Mexico, Morocco, Nepal, Netherlands, New Zealand, Nigeria, Pakistan, Peru, Philippines, Poland, Romania, Russia, Saudi Arabia, Singapore, South Africa, South Korea, Spain, Taiwan, Thailand, Turkey, Ukraine, United Kingdom, United States, Vietnam, Zimbabwe

Tasks:

* cultural_bench_easy_<country>
* cultural_bench_hard_<country>


### Checklist

For adding novel benchmarks/datasets to the library:

* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
