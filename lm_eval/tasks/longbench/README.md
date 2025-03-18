# Task-name

### Paper

Title: `LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks`

Abstract: `This paper introduces LongBench v2, a benchmark designed to assess the ability of LLMs to handle long-context problems requiring deep understanding and reasoning across real-world multitasks. LongBench v2 consists of 503 challenging multiple-choice questions, with contexts ranging from 8k to 2M words, across six major task categories: single-document QA, multi-document QA, long in-context learning, long-dialogue history understanding, code repository understanding, and long structured data understanding.`

Homepage: `https://github.com/THUDM/LongBench`


### Citation

```
@article{bai2024longbench2,
  title={LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks},
  author={Yushi Bai and Shangqing Tu and Jiajie Zhang and Hao Peng and Xiaozhi Wang and Xin Lv and Shulin Cao and Jiazheng Xu and Lei Hou and Yuxiao Dong and Jie Tang and Juanzi Li},
  journal={arXiv preprint arXiv:2412.15204},
  year={2024}
}
@inproceedings{bai2024longbench,
    title = "{L}ong{B}ench: A Bilingual, Multitask Benchmark for Long Context Understanding",
    author = "Bai, Yushi and Lv, Xin  and Zhang, Jiajie  and Lyu, Hongchang  and
      Tang, Jiankai  and Huang, Zhidian  and Du, Zhengxiao  and Liu, Xiao  and Zeng, Aohan  and Hou, Lei  and Dong, Yuxiao  and Tang, Jie  and Li, Juanzi",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.172",
    doi = "10.18653/v1/2024.acl-long.172",
    pages = "3119--3137",
}
```

### Groups, Tags, and Tasks

#### Groups

[//]: # (* `group_name`: `Short description`)

#### Tags

* `LongBench`: `Benchmark with 21 tasks (avg. 5k-15k tokens) for evaluating long-context capabilities`
* `LongBench-E`: `Modified version with uniform length distribution (0-4k, 4k-8k, 8k+) for analyzing performance across different input lengths`

#### Tasks

* `2wikimqa`: `Question answering task using multiple Wikipedia articles as reference`
* `2wikimqa_e`: `Extended version of 2wikimqa with additional complexity or data`
* `dureader`: `Chinese machine reading comprehension dataset with real-world queries`
* `gov_report`: `Summarization task for long government reports and documents`
* `gov_report_e`: `Extended version of gov_report with additional complexity or data`
* `hotpotqa`: `Multi-hop question answering requiring reasoning across multiple paragraphs`
* `hotpotqa_e`: `Extended version of hotpotqa with additional complexity or data`
* `lcc`: `Long-form content classification across various categories and domains`
* `lcc_e`: `Extended version of lcc with additional complexity or data`
* `lsht`: `Large-scale hierarchical text classification task`
* `multi_news`: `Multi-document news summarization task`
* `multi_news_e`: `Extended version of multi_news with additional complexity or data`
* `multifieldqa_en`: `English question answering across multiple knowledge domains or fields`
* `multifieldqa_en_e`: `Extended version of multifieldqa_en with additional complexity or data`
* `multifieldqa_zh`: `Chinese question answering across multiple knowledge domains or fields`
* `musique`: `Multi-step reasoning question answering with complex queries`
* `narrativeqa`: `Question answering based on book and movie narratives`
* `passage_count`: `Task requiring counting or quantifying information across passages`
* `passage_count_e`: `Extended version of passage_count with additional complexity or data`
* `passage_retrieval_en`: `English passage retrieval task for information seeking`
* `passage_retrieval_en_e`: `Extended version of passage_retrieval_en with additional complexity or data`
* `passage_retrieval_zh`: `Chinese passage retrieval task for information seeking`
* `qasper`: `Question answering on scientific papers requiring domain knowledge`
* `qasper_e`: `Extended version of qasper with additional complexity or data`
* `qmsum`: `Query-based meeting summarization task`
* `repobench-p`: `Programming task based on code repositories`
* `repobench-p_e`: `Extended version of repobench-p with additional complexity or data`
* `samsum`: `Dialogue summarization for messenger-like conversations`
* `samsum_e`: `Extended version of samsum with additional complexity or data`
* `trec`: `Question classification task for information retrieval`
* `trec_e`: `Extended version of trec with additional complexity or data`
* `triviaqa`: `Large-scale question answering dataset with trivia questions`
* `triviaqa_e`: `Extended version of triviaqa with additional complexity or data`
* `vcsum`: `Video conference summarization task`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
