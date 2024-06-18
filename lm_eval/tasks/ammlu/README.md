# ArabicMMLU

### Paper

Title: ArabicMMLU: Assessing Massive Multitask Language Understanding in Arabic

Abstract: https://arxiv.org/abs/2402.12840

The focus of language model evaluation has
transitioned towards reasoning and knowledge intensive tasks, driven by advancements in pretraining large models. While state-of-the-art models are partially trained on large Arabic texts, evaluating their performance in Arabic remains challenging due to the limited availability of relevant datasets. To bridge this gap, we present ArabicMMLU, the first multi-task language understanding benchmark for Arabic language, sourced from school exams across diverse educational levels in different countries spanning North Africa, the Levant, and the Gulf regions. Our data comprises 40 tasks and 14,575 multiple-choice questions in Modern Standard Arabic (MSA), and is carefully constructed by collaborating with native speakers in the region. Our comprehensive evaluations of 35 models reveal substantial room for improvement, particularly among the best open-source models. Notably, BLOOMZ, mT0, LLama2, and Falcon struggle to achieve a score of 50%, while even the top-performing Arabic centric model only achieves a score of 62.3%.

Homepage: https://github.com/mbzuai-nlp/ArabicMMLU


### Citation

```
@misc{koto2024arabicmmlu,
      title={ArabicMMLU: Assessing Massive Multitask Language Understanding in Arabic}, 
      author={Fajri Koto and Haonan Li and Sara Shatnawi and Jad Doughman and Abdelrahman Boda Sadallah and Aisha Alraeesi and Khalid Almubarak and Zaid Alyafeai and Neha Sengupta and Shady Shehata and Nizar Habash and Preslav Nakov and Timothy Baldwin},
      year={2024},
      eprint={2402.12840},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```

### Groups and Tasks

#### Groups

* `ammlu_stem`: evaluates STEM ArabicMMLU tasks.
* `ammlu_social_science`: evaluates social science ArabicMMLU tasks.
* `ammlu_humanities`: evaluates humanities ArabicMMLU tasks.
* `ammlu_language`: evaluates Arabic language ArabicMMLU tasks.
* `ammlu_other`: evaluates other ArabicMMLU tasks.

#### Tasks

* `ammlu`: evaluates all ArabicMMLU tasks.
