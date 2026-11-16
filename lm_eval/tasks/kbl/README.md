# kbl

### Paper

Title: `Developing a Pragmatic Benchmark for Assessing Korean Legal Language Understanding in Large Language Models`

Abstract: `Large language models (LLMs) have demonstrated remarkable performance in the legal domain, with GPT-4 even passing the Uniform Bar Exam in the U.S. However their efficacy remains limited for non-standardized tasks and tasks in languages other than English. This underscores the need for careful evaluation of LLMs within each legal system before application. Here, we introduce KBL, a benchmark for assessing the Korean legal language understanding of LLMs, consisting of (1) 7 legal knowledge tasks (510 examples), (2) 4 legal reasoning tasks (288 examples), and (3) the Korean bar exam (4 domains, 53 tasks, 2,510 examples). First two datasets were developed in close collaboration with lawyers to evaluate LLMs in practical scenarios in a certified manner. Furthermore, considering legal practitioners' frequent use of extensive legal documents for research, we assess LLMs in both a closed book setting, where they rely solely on internal knowledge, and a retrieval-augmented generation (RAG) setting, using a corpus of Korean statutes and precedents. The results indicate substantial room and opportunities for improvement.`

`Korean Benchmark for Legal Language Understanding`

Homepage: `https://github.com/lbox-kr/kbl`


### Citation

```
@inproceedings{kim2024kbl,
    title = "Developing a Pragmatic Benchmark for Assessing {K}orean Legal Language Understanding in Large Language Models",
    author = {Yeeun Kim and Young Rok Choi and Eunkyung Choi and Jinhwan Choi and Hai Jin Park and Wonseok Hwang},
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.319",
    pages = "5573--5595",
}
```

### Groups, Tags, and Tasks

#### Groups

#### Tags

* `kbl`: `All kbl tasks (7 knowledge, 4 reasoning, and 39 bar exam)`
* `kbl_knowledge_em`: `7 knowledge tasks`
* `kbl_reasoning_em`: `4 reasoning tasks`
* `kbl_bar_exam_em`: `53 bar exam tasks`
* `kbl_bar_exam_em_civil`: `13 bar exam tasks, civil law`
* `kbl_bar_exam_em_criminal`: `13 bar exam tasks,  criminal law`
* `kbl_bar_exam_em_public`: `13 bar exam tasks, public law`
* `kbl_bar_exam_em_responsibility`: `14 bar exam tasks, professional responsibility (RESP) examination`


#### Tasks

* `kbl_common_legal_mistake_qa_em`: `A QA task evaluating common legal misconceptions from the general public.`
* `kbl_knowledge_common_legal_mistake_qa_reasoning`: `Similar to 'kbl_common_legal_mistake_qa_em' but the answers are presented with correct/wrong rationals.`
* `kbl_knowledge_legal_concept_qa`: `A QA task addressing knowledge about complex legal concepts (legal terms).`
* `kbl_knowledge_offense_component_qa`: `A QA task evaluating whether a model knows specific actions meet the actual elements of a criminal offense.`
* `kbl_knowledge_query_and_statute_matching_qa`: `A QA task assessing whether the language model can accurately identify the relevant statute for a given query.`
* `kbl_knowledge_statute_hallucination_qa`: `A QA task evaluating whether a model can select the correct answer consists of a pair of (fictitious) statute and corresponding reasoning for given confusing legal questions.`
* `kbl_knowledge_statute_number_and_content_matching_qa`: `A QA dataset for evaluating where a model can accurately match the content of a law to its specific statute number.`
* `kbl_reasoning_case_relevance_qa_p`: `A QA task where a model needs to determine whether a given precedent is relavent to an input precedent.`
* `kbl_reasoning_case_relevance_qa_q`: `A QA task where a model needs to determine whether a given precedent is relavent to an input query.`
* `kbl_reasoning_causal_reasoning_qa`: `A QA task where a model needs to assess whether the defendant’s actions were the direct and decisive cause of the victim’s injury or death for each given factual description and claims.`
* `kbl_reasoning_statement_consistency_qa`: `A QA task where a model is required to accurately determine whether two presented statements are consistent with each other.`
* `bar_exam_civil_2012`: `Korean bar exam multiple-choice questions, civil law`
* `bar_exam_civil_2013`: `Korean bar exam multiple-choice questions, civil law`
* `bar_exam_civil_2014`: `Korean bar exam multiple-choice questions, civil law`
* `bar_exam_civil_2015`: `Korean bar exam multiple-choice questions, civil law`
* `bar_exam_civil_2016`: `Korean bar exam multiple-choice questions, civil law`
* `bar_exam_civil_2017`: `Korean bar exam multiple-choice questions, civil law`
* `bar_exam_civil_2018`: `Korean bar exam multiple-choice questions, civil law`
* `bar_exam_civil_2019`: `Korean bar exam multiple-choice questions, civil law`
* `bar_exam_civil_2020`: `Korean bar exam multiple-choice questions, civil law`
* `bar_exam_civil_2021`: `Korean bar exam multiple-choice questions, civil law`
* `bar_exam_civil_2022`: `Korean bar exam multiple-choice questions, civil law`
* `bar_exam_civil_2023`: `Korean bar exam multiple-choice questions, civil law`
* `bar_exam_civil_2024`: `Korean bar exam multiple-choice questions, civil law`
* `bar_exam_criminal_2012`: `Korean bar exam multiple-choice questions, criminal law`
* `bar_exam_criminal_2013`: `Korean bar exam multiple-choice questions, criminal law`
* `bar_exam_criminal_2014`: `Korean bar exam multiple-choice questions, criminal law`
* `bar_exam_criminal_2015`: `Korean bar exam multiple-choice questions, criminal law`
* `bar_exam_criminal_2016`: `Korean bar exam multiple-choice questions, criminal law`
* `bar_exam_criminal_2017`: `Korean bar exam multiple-choice questions, criminal law`
* `bar_exam_criminal_2018`: `Korean bar exam multiple-choice questions, criminal law`
* `bar_exam_criminal_2019`: `Korean bar exam multiple-choice questions, criminal law`
* `bar_exam_criminal_2020`: `Korean bar exam multiple-choice questions, criminal law`
* `bar_exam_criminal_2021`: `Korean bar exam multiple-choice questions, criminal law`
* `bar_exam_criminal_2022`: `Korean bar exam multiple-choice questions, criminal law`
* `bar_exam_criminal_2023`: `Korean bar exam multiple-choice questions, criminal law`
* `bar_exam_criminal_2024`: `Korean bar exam multiple-choice questions, criminal law`
* `bar_exam_public_2012`: `Korean bar exam multiple-choice questions, public law`
* `bar_exam_public_2013`: `Korean bar exam multiple-choice questions, public law`
* `bar_exam_public_2014`: `Korean bar exam multiple-choice questions, public law`
* `bar_exam_public_2015`: `Korean bar exam multiple-choice questions, public law`
* `bar_exam_public_2016`: `Korean bar exam multiple-choice questions, public law`
* `bar_exam_public_2017`: `Korean bar exam multiple-choice questions, public law`
* `bar_exam_public_2018`: `Korean bar exam multiple-choice questions, public law`
* `bar_exam_public_2019`: `Korean bar exam multiple-choice questions, public law`
* `bar_exam_public_2020`: `Korean bar exam multiple-choice questions, public law`
* `bar_exam_public_2021`: `Korean bar exam multiple-choice questions, public law`
* `bar_exam_public_2022`: `Korean bar exam multiple-choice questions, public law`
* `bar_exam_public_2023`: `Korean bar exam multiple-choice questions, public law`
* `bar_exam_public_2024`: `Korean bar exam multiple-choice questions, public law`
* `bar_exam_responsibility_2010`: `Korean bar exam multiple-choice questions, professional responsibility (RESP) examination`
* `bar_exam_responsibility_2011`: `Korean bar exam multiple-choice questions, professional responsibility (RESP) examination`
* `bar_exam_responsibility_2012`: `Korean bar exam multiple-choice questions, professional responsibility (RESP) examination`
* `bar_exam_responsibility_2013`: `Korean bar exam multiple-choice questions, professional responsibility (RESP) examination`
* `bar_exam_responsibility_2014`: `Korean bar exam multiple-choice questions, professional responsibility (RESP) examination`
* `bar_exam_responsibility_2015`: `Korean bar exam multiple-choice questions, professional responsibility (RESP) examination`
* `bar_exam_responsibility_2016`: `Korean bar exam multiple-choice questions, professional responsibility (RESP) examination`
* `bar_exam_responsibility_2017`: `Korean bar exam multiple-choice questions, professional responsibility (RESP) examination`
* `bar_exam_responsibility_2018`: `Korean bar exam multiple-choice questions, professional responsibility (RESP) examination`
* `bar_exam_responsibility_2019`: `Korean bar exam multiple-choice questions, professional responsibility (RESP) examination`
* `bar_exam_responsibility_2020`: `Korean bar exam multiple-choice questions, professional responsibility (RESP) examination`
* `bar_exam_responsibility_2021`: `Korean bar exam multiple-choice questions, professional responsibility (RESP) examination`
* `bar_exam_responsibility_2022`: `Korean bar exam multiple-choice questions, professional responsibility (RESP) examination`
* `bar_exam_responsibility_2023`: `Korean bar exam multiple-choice questions, professional responsibility (RESP) examination`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
