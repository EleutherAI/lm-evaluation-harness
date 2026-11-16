# Arabic Leaderboard


Title: Open Arabic LLM Leaderboard

The Open Arabic LLM Leaderboard evaluates language models on a large number of different evaluation tasks that reflect the characteristics of the Arabic language and culture.
The benchmark uses several datasets, most of them translated to Arabic, and validated by native Arabic speakers. They also used benchmarks from other papers or prepared benchmarks from scratch natively for Arabic.

Homepage: https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard

### Citation

```

@misc{OALL,
  author = {Elfilali, Ali and Alobeidli, Hamza and Fourrier, Clémentine and Boussaha, Basma El Amel and Cojocaru, Ruxandra and Habib, Nathan and Hacid, Hakim},
  title = {Open Arabic LLM Leaderboard},
  year = {2024},
  publisher = {OALL},
  howpublished = "\url{https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard}"
}

@inproceedings{almazrouei-etal-2023-alghafa,
    title = "{A}l{G}hafa Evaluation Benchmark for {A}rabic Language Models",
    author = "Almazrouei, Ebtesam  and
      Cojocaru, Ruxandra  and
      Baldo, Michele  and
      Malartic, Quentin  and
      Alobeidli, Hamza  and
      Mazzotta, Daniele  and
      Penedo, Guilherme  and
      Campesan, Giulia  and
      Farooq, Mugariya  and
      Alhammadi, Maitha  and
      Launay, Julien  and
      Noune, Badreddine",
    editor = "Sawaf, Hassan  and
      El-Beltagy, Samhaa  and
      Zaghouani, Wajdi  and
      Magdy, Walid  and
      Abdelali, Ahmed  and
      Tomeh, Nadi  and
      Abu Farha, Ibrahim  and
      Habash, Nizar  and
      Khalifa, Salam  and
      Keleg, Amr  and
      Haddad, Hatem  and
      Zitouni, Imed  and
      Mrini, Khalil  and
      Almatham, Rawan",
    booktitle = "Proceedings of ArabicNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.arabicnlp-1.21",
    doi = "10.18653/v1/2023.arabicnlp-1.21",
    pages = "244--275",
    abstract = "Recent advances in the space of Arabic large language models have opened up a wealth of potential practical applications. From optimal training strategies, large scale data acquisition and continuously increasing NLP resources, the Arabic LLM landscape has improved in a very short span of time, despite being plagued by training data scarcity and limited evaluation resources compared to English. In line with contributing towards this ever-growing field, we introduce AlGhafa, a new multiple-choice evaluation benchmark for Arabic LLMs. For showcasing purposes, we train a new suite of models, including a 14 billion parameter model, the largest monolingual Arabic decoder-only model to date. We use a collection of publicly available datasets, as well as a newly introduced HandMade dataset consisting of 8 billion tokens. Finally, we explore the quantitative and qualitative toxicity of several Arabic models, comparing our models to existing public Arabic LLMs.",
}
@misc{huang2023acegpt,
      title={AceGPT, Localizing Large Language Models in Arabic},
      author={Huang Huang and Fei Yu and Jianqing Zhu and Xuening Sun and Hao Cheng and Dingjie Song and Zhihong Chen and Abdulmohsen Alharthi and Bang An and Ziche Liu and Zhiyi Zhang and Junying Chen and Jianquan Li and Benyou Wang and Lian Zhang and Ruoyu Sun and Xiang Wan and Haizhou Li and Jinchao Xu},
      year={2023},
      eprint={2309.12053},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{lighteval,
  author = {Fourrier, Clémentine and Habib, Nathan and Wolf, Thomas and Tunstall, Lewis},
  title = {LightEval: A lightweight framework for LLM evaluation},
  year = {2023},
  version = {0.3.0},
  url = {https://github.com/huggingface/lighteval}
}
```

### Groups and Tasks

* `arabic_leaderboard_alghafa`: A multiple-choice evaluation benchmark for zero- and few-shot evaluation of Arabic LLMs prepared from scratch natively for Arabic.
    * Paper: https://aclanthology.org/2023.arabicnlp-1.21.pdf
    * You can find the list of the tasks as follows:
      * `arabic_leaderboard_alghafa_mcq_exams_test_ar`
      * `arabic_leaderboard_alghafa_meta_ar_dialects`
      * `arabic_leaderboard_alghafa_meta_ar_msa`
      * `arabic_leaderboard_alghafa_multiple_choice_facts_truefalse_balanced_task`
      * `arabic_leaderboard_alghafa_multiple_choice_grounded_statement_soqal_task`
      * `arabic_leaderboard_alghafa_multiple_choice_grounded_statement_xglue_mlqa_task`
      * `arabic_leaderboard_alghafa_multiple_choice_rating_sentiment_no_neutral_task`
      * `arabic_leaderboard_alghafa_multiple_choice_rating_sentiment_task`
      * `arabic_leaderboard_alghafa_multiple_choice_sentiment_task`
* `arabic_leaderboard_arabic_exams`: A question answering benchmark for high school examinations in different school subjects that requires knowledge and reasoning in different languages in multiple domains.
    * Paper: https://aclanthology.org/2020.emnlp-main.438.pdf
* `arabic_leaderboard_arabic_mmlu`: A multi-task language understanding benchmark for the Arabic language, sourced from school exams across diverse educational levels in different countries with native speakers in the region.
  The data comprises multiple choice questions in 40 tasks.
    * Paper: https://arxiv.org/pdf/2402.12840
    * You can find the list of the tasks as follows:
      * `arabic_leaderboard_arabic_mmlu_abstract_algebra`
      * `arabic_leaderboard_arabic_mmlu_anatomy`
      * `arabic_leaderboard_arabic_mmlu_astronomy`
      * `arabic_leaderboard_arabic_mmlu_business_ethics`
      * `arabic_leaderboard_arabic_mmlu_clinical_knowledge`
      * `arabic_leaderboard_arabic_mmlu_college_biology`
      * `arabic_leaderboard_arabic_mmlu_college_chemistry`
      * `arabic_leaderboard_arabic_mmlu_college_computer_science`
      * `arabic_leaderboard_arabic_mmlu_college_mathematics`
      * `arabic_leaderboard_arabic_mmlu_college_medicine`
      * `arabic_leaderboard_arabic_mmlu_college_physics`
      * `arabic_leaderboard_arabic_mmlu_computer_security`
      * `arabic_leaderboard_arabic_mmlu_conceptual_physics`
      * `arabic_leaderboard_arabic_mmlu_econometrics`
      * `arabic_leaderboard_arabic_mmlu_electrical_engineering`
      * `arabic_leaderboard_arabic_mmlu_elementary_mathematics`
      * `arabic_leaderboard_arabic_mmlu_formal_logic`
      * `arabic_leaderboard_arabic_mmlu_global_facts`
      * `arabic_leaderboard_arabic_mmlu_high_school_biology`
      * `arabic_leaderboard_arabic_mmlu_high_school_chemistry`
      * `arabic_leaderboard_arabic_mmlu_high_school_computer_science`
      * `arabic_leaderboard_arabic_mmlu_high_school_european_history`
      * `arabic_leaderboard_arabic_mmlu_high_school_geography`
      * `arabic_leaderboard_arabic_mmlu_high_school_government_and_politics`
      * `arabic_leaderboard_arabic_mmlu_high_school_macroeconomics`
      * `arabic_leaderboard_arabic_mmlu_high_school_mathematics`
      * `arabic_leaderboard_arabic_mmlu_high_school_microeconomics`
      * `arabic_leaderboard_arabic_mmlu_high_school_physics`
      * `arabic_leaderboard_arabic_mmlu_high_school_psychology`
      * `arabic_leaderboard_arabic_mmlu_high_school_statistics`
      * `arabic_leaderboard_arabic_mmlu_high_school_us_history`
      * `arabic_leaderboard_arabic_mmlu_high_school_us_history`
      * `arabic_leaderboard_arabic_mmlu_human_aging`
      * `arabic_leaderboard_arabic_mmlu_human_sexuality`
      * `arabic_leaderboard_arabic_mmlu_international_law`
      * `arabic_leaderboard_arabic_mmlu_jurisprudence`
      * `arabic_leaderboard_arabic_mmlu_logical_fallacies`
      * `arabic_leaderboard_arabic_mmlu_machine_learning`
      * `arabic_leaderboard_arabic_mmlu_management`
      * `arabic_leaderboard_arabic_mmlu_marketing`
      * `arabic_leaderboard_arabic_mmlu_medical_genetics`
      * `arabic_leaderboard_arabic_mmlu_miscellaneous`
      * `arabic_leaderboard_arabic_mmlu_moral_disputes`
      * `arabic_leaderboard_arabic_mmlu_moral_scenarios`
      * `arabic_leaderboard_arabic_mmlu_nutrition`
      * `arabic_leaderboard_arabic_mmlu_philosophy`
      * `arabic_leaderboard_arabic_mmlu_prehistory`
      * `arabic_leaderboard_arabic_mmlu_professional_accounting`
      * `arabic_leaderboard_arabic_mmlu_professional_law`
      * `arabic_leaderboard_arabic_mmlu_professional_medicine`
      * `arabic_leaderboard_arabic_mmlu_professional_psychology`
      * `arabic_leaderboard_arabic_mmlu_public_relations`
      * `arabic_leaderboard_arabic_mmlu_security_studies`
      * `arabic_leaderboard_arabic_mmlu_sociology`
      * `arabic_leaderboard_arabic_mmlu_us_foreign_policy`
      * `arabic_leaderboard_arabic_mmlu_virology`
      * `arabic_leaderboard_arabic_mmlu_world_religions`
* `arabic_leaderboard_arabic_mt_arc_challenge`: AI2 Reasoning Challenge (ARC) is a multiple-choice question task. The dataset contains only natural, grade-school science questions,
  written for human tests. The challenge set contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurence algorithm. (machine translated benchmark - part of the Alghafa Arabic translated LLM benchmark)
    * Paper: https://aclanthology.org/2023.arabicnlp-1.21.pdf
* `arabic_leaderboard_arabic_mt_arc_easy`: This dataset is the same as `arabic_arc_challenge`, except it is not from the challenge set.
    * Paper: https://aclanthology.org/2023.arabicnlp-1.21.pdf
* `arabic_leaderboard_arabic_mt_boolq`: A true/false questions dataset that contains the columns passage, question, and the answer (i.e., true/false). (machine translated benchmark - part of the Alghafa Arabic translated LLM benchmark)
    * Paper: https://aclanthology.org/2023.arabicnlp-1.21.pdf
* `arabic_leaderboard_arabic_mt_copa`: Choice Of Plausible Alternatives (COPA) is a multiple-choice question dataset, which involves open-domain commonsense causal reasoning. (machine translated benchmark - part of the Alghafa Arabic translated LLM benchmark)
    * Paper: https://aclanthology.org/2023.arabicnlp-1.21.pdf
* `arabic_leaderboard_arabic_mt_hellaswag`: The tesk is to choose the next set of sentences, based on the given candidates. The tasks involve reading comprehension and information retrieval challenges
  by testing the abilities of the models on basic knowledge (i.e., from 3rd grade to 9th) and commonsense inference. (machine translated benchmark - part of the Alghafa Arabic translated LLM benchmark)
    * Paper: https://aclanthology.org/2023.arabicnlp-1.21.pdf
* `arabic_leaderboard_arabic_mt_mmlu`: A multiple-choice question answering dataset from various branches of knowledge including humanities, social sciences, hard sciences, and other areas. The examples in the English dataset are translated into Arabic using ChatGPT with a translation prompt.
    * Paper: https://aclanthology.org/2023.arabicnlp-1.21.pdf
* `arabic_leaderboard_arabic_mt_openbook_qa`: A multiple-choice openbook question answering dataset that requires external knowledge and reasoning. The open book that comes with these questions is
  based on elementary level science facts. (machine translated benchmark - part of the Alghafa Arabic translated LLM benchmark)
    * Paper: https://aclanthology.org/2023.arabicnlp-1.21.pdf
* `arabic_leaderboard_arabic_mt_piqa`: Physical Interaction Question Answering (PIQA) is a multiple-choice question answering based on physical commonsense reasoning. (machine translated benchmark - part of the Alghafa Arabic translated LLM benchmark)
    * Paper: https://aclanthology.org/2023.arabicnlp-1.21.pdf
* `arabic_leaderboard_arabic_mt_race`: A multiple-choice questions dataset to assess reading comprehension tasks based on English exams in China - designed for middle school and high school students
  (machine translated benchmark - part of the Alghafa Arabic translated LLM benchmark)
    * Paper: https://aclanthology.org/2023.arabicnlp-1.21.pdf
* `arabic_leaderboard_arabic_mt_sciq`: A multiple-choice Science Question Answering task to assess understanding of scientific concepts about physics, chemistry, and biology. (machine translated benchmark - part of the Alghafa Arabic translated LLM benchmark)
    * Paper: https://aclanthology.org/2023.arabicnlp-1.21.pdf
* `arabic_leaderboard_arabic_mt_toxigen`: This benchmark consists of tasks designed to evaluate language models and classify input text as hateful or not hateful. (machine translated benchmark - part of the Alghafa Arabic translated LLM benchmark)
    * Paper: https://aclanthology.org/2023.arabicnlp-1.21.pdf
* `arabic_leaderboard_acva`: Arabic-Culture-Value-Alignment (ACVA) is a yes/no question dataset, generated by GPT3.5 Turbo from Arabic topics to assess model alignment with Arabic values and cultures.
    * Paper: https://arxiv.org/pdf/2309.12053
    * You can find the list of the tasks as follows:
      - `arabic_leaderboard_acva_Algeria`
      - `arabic_leaderboard_acva_Ancient_Egypt`
      - `arabic_leaderboard_acva_Arab_Empire`
      - `arabic_leaderboard_acva_Arabic_Architecture`
      - `arabic_leaderboard_acva_Arabic_Art`
      - `arabic_leaderboard_acva_Arabic_Astronomy`
      - `arabic_leaderboard_acva_Arabic_Calligraphy`
      - `arabic_leaderboard_acva_Arabic_Ceremony`
      - `arabic_leaderboard_acva_Arabic_Clothing`
      - `arabic_leaderboard_acva_Arabic_Culture`
      - `arabic_leaderboard_acva_Arabic_Food`
      - `arabic_leaderboard_acva_Arabic_Funeral`
      - `arabic_leaderboard_acva_Arabic_Geography`
      - `arabic_leaderboard_acva_Arabic_History`
      - `arabic_leaderboard_acva_Arabic_Language_Origin`
      - `arabic_leaderboard_acva_Arabic_Literature`
      - `arabic_leaderboard_acva_Arabic_Math`
      - `arabic_leaderboard_acva_Arabic_Medicine`
      - `arabic_leaderboard_acva_Arabic_Music`
      - `arabic_leaderboard_acva_Arabic_Ornament`
      - `arabic_leaderboard_acva_Arabic_Philosophy`
      - `arabic_leaderboard_acva_Arabic_Physics_and_Chemistry`
      - `arabic_leaderboard_acva_Arabic_Wedding`
      - `arabic_leaderboard_acva_Bahrain`
      - `arabic_leaderboard_acva_Comoros`
      - `arabic_leaderboard_acva_Egypt_modern`
      - `arabic_leaderboard_acva_InfluenceFromAncientEgypt`
      - `arabic_leaderboard_acva_InfluenceFromByzantium`
      - `arabic_leaderboard_acva_InfluenceFromChina`
      - `arabic_leaderboard_acva_InfluenceFromGreece`
      - `arabic_leaderboard_acva_InfluenceFromIslam`
      - `arabic_leaderboard_acva_InfluenceFromPersia`
      - `arabic_leaderboard_acva_InfluenceFromRome`
      - `arabic_leaderboard_acva_Iraq`
      - `arabic_leaderboard_acva_Islam_Education`
      - `arabic_leaderboard_acva_Islam_branches_and_schools`
      - `arabic_leaderboard_acva_Islamic_law_system`
      - `arabic_leaderboard_acva_Jordan`
      - `arabic_leaderboard_acva_Kuwait`
      - `arabic_leaderboard_acva_Lebanon`
      - `arabic_leaderboard_acva_Libya`
      - `arabic_leaderboard_acva_Mauritania`
      - `arabic_acva_Mesopotamia_civilization`
      - `arabic_leaderboard_acva_Morocco`
      - `arabic_leaderboard_acva_Oman`
      - `arabic_leaderboard_acva_Palestine`
      - `arabic_leaderboard_acva_Qatar`
      - `arabic_leaderboard_acva_Saudi_Arabia`
      - `arabic_leaderboard_acva_Somalia`
      - `arabic_leaderboard_acva_Sudan`
      - `arabic_leaderboard_acva_Syria`
      - `arabic_leaderboard_acva_Tunisia`
      - `arabic_leaderboard_acva_United_Arab_Emirates`
      - `arabic_leaderboard_acva_Yemen`
      - `arabic_leaderboard_acva_communication`
      - `arabic_leaderboard_acva_computer_and_phone`
      - `arabic_leaderboard_acva_daily_life`
      - `arabic_leaderboard_acva_entertainment`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
