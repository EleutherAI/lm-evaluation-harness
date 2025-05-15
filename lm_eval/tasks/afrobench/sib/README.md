#

## Paper
Title: `SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects`

Paper Link: https://aclanthology.org/2024.eacl-long.14/

## Abstract
>Despite the progress in building multilingual language models, evaluation is often limited to a few languages with available datasets which excludes a large number of low-resource languages. In this paper, we create SIB-200—a large-scale open-sourced benchmark dataset for topic classification in 205 languages and dialects to address the lack of evaluation dataset for Natural Language Understanding (NLU). For many of the languages covered in SIB-200, this is the first publicly available evaluation dataset for NLU. The dataset is based on Flores-200 machine translation corpus. We annotated the English portion of the dataset and extended the sentence-level annotation to the remaining 204 languages covered in the corpus. Despite the simplicity of this task, our evaluation in full-supervised setting, cross-lingual transfer setting and prompting of large language model setting show that there is still a large gap between the performance of high-resource and low-resource languages when multilingual evaluation is scaled to numerous world languages. We found that languages unseen during the pre-training of multilingual language models, languages from under-represented families (like Nilotic and Altantic-Congo), and languages from the regions of Africa, Americas, Oceania and South East Asia, often have the lowest performance on our topic classification dataset. We hope our dataset %will encourages a more inclusive evaluation of multilingual language models on a more diverse set of languages.

HomePage: https://github.com/dadelani/sib-200

### Citation

```
@inproceedings{adelani-etal-2024-sib,
    title = "{SIB}-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects",
    author = "Adelani, David Ifeoluwa  and
      Liu, Hannah  and
      Shen, Xiaoyu  and
      Vassilyev, Nikita  and
      Alabi, Jesujoba O.  and
      Mao, Yanke  and
      Gao, Haonan  and
      Lee, En-Shiun Annie",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.14/",
    pages = "226--245",
    abstract = "Despite the progress in building multilingual language models, evaluation is often limited to a few languages with available datasets which excludes a large number of low-resource languages. In this paper, we create SIB-200{---}a large-scale open-sourced benchmark dataset for topic classification in 205 languages and dialects to address the lack of evaluation dataset for Natural Language Understanding (NLU). For many of the languages covered in SIB-200, this is the first publicly available evaluation dataset for NLU. The dataset is based on Flores-200 machine translation corpus. We annotated the English portion of the dataset and extended the sentence-level annotation to the remaining 204 languages covered in the corpus. Despite the simplicity of this task, our evaluation in full-supervised setting, cross-lingual transfer setting and prompting of large language model setting show that there is still a large gap between the performance of high-resource and low-resource languages when multilingual evaluation is scaled to numerous world languages. We found that languages unseen during the pre-training of multilingual language models, languages from under-represented families (like Nilotic and Altantic-Congo), and languages from the regions of Africa, Americas, Oceania and South East Asia, often have the lowest performance on our topic classification dataset. We hope our dataset {\%}will encourages a more inclusive evaluation of multilingual language models on a more diverse set of languages."
}
```
