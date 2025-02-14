# Global-MMLU

### Paper

Title: `Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation`

Abstract: [https://arxiv.org/abs/2412.03304](https://arxiv.org/abs/2412.03304)

Global-MMLU 🌍 is a multilingual evaluation set spanning 42 languages, including English. This dataset combines machine translations for MMLU questions along with professional translations and crowd-sourced post-edits. It also includes cultural sensitivity annotations for a subset of the questions (2850 questions per language) and classifies them as Culturally Sensitive (CS) 🗽 or Culturally Agnostic (CA) ⚖️. These annotations were collected as part of an open science initiative led by Cohere For AI in collaboration with many external collaborators from both industry and academia.

Global-MMLU-Lite is a balanced collection of culturally sensitive and culturally agnostic MMLU tasks. It is designed for efficient evaluation of multilingual models in 15 languages (including English). Only languages with human translations and post-edits in the original [Global-MMLU](https://huggingface.co/datasets/CohereForAI/Global-MMLU) 🌍 dataset have been included in the lite version.

Homepage: \
[https://huggingface.co/datasets/CohereForAI/Global-MMLU](https://huggingface.co/datasets/CohereForAI/Global-MMLU) \
[https://huggingface.co/datasets/CohereForAI/Global-MMLU-Lite](https://huggingface.co/datasets/CohereForAI/Global-MMLU-Lite)


#### Groups

* `global_mmlu_{lang}`: This group uses `Global-MMLU-Lite` benchmark which supports 14 languages.
* `global_mmlu_full_{lang}`: This group uses `Global-MMLU` benchmark which supports 42 languages.

#### Subgroups (support only for `full` version)

* `global_mmlu_full_stem`
* `global_mmlu_full_humanities`
* `global_mmlu_full_social_sciences`
* `global_mmlu_full_other`

### Citation

```bibtex
@misc{singh2024globalmmluunderstandingaddressing,
      title={Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation},
      author={Shivalika Singh and Angelika Romanou and Clémentine Fourrier and David I. Adelani and Jian Gang Ngui and Daniel Vila-Suero and Peerat Limkonchotiwat and Kelly Marchisio and Wei Qi Leong and Yosephine Susanto and Raymond Ng and Shayne Longpre and Wei-Yin Ko and Madeline Smith and Antoine Bosselut and Alice Oh and Andre F. T. Martins and Leshem Choshen and Daphne Ippolito and Enzo Ferrante and Marzieh Fadaee and Beyza Ermis and Sara Hooker},
      year={2024},
      eprint={2412.03304},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.03304},
}
```
