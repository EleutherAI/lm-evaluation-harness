# LongBench v2

### Paper

Title: `LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks`

Abstract: `This paper introduces LongBench v2, a benchmark designed to assess the ability of LLMs to handle long-context problems requiring deep understanding and reasoning across real-world multitasks. LongBench v2 consists of 503 challenging multiple-choice questions, with contexts ranging from 8k to 2M words, across six major task categories: single-document QA, multi-document QA, long in-context learning, long-dialogue history understanding, code repository understanding, and long structured data understanding. To ensure the breadth and the practicality, we collect data from nearly 100 highly educated individuals with diverse professional backgrounds. We employ both automated and manual review processes to maintain high quality and difficulty, resulting in human experts achieving only 53.7% accuracy under a 15-minute time constraint. Our evaluation reveals that the best-performing model, when directly answers the questions, achieves only 50.1% accuracy. In contrast, the o1-preview model, which includes longer reasoning, achieves 57.7%, surpassing the human baseline by 4%. These results highlight the importance of enhanced reasoning ability and scaling inference-time compute to tackle the long-context challenges in LongBench v2.`

Homepage: `https://github.com/THUDM/LongBench`


### Citation

```
@article{bai2024longbench2,
  title={LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks},
  author={Yushi Bai and Shangqing Tu and Jiajie Zhang and Hao Peng and Xiaozhi Wang and Xin Lv and Shulin Cao and Jiazheng Xu and Lei Hou and Yuxiao Dong and Jie Tang and Juanzi Li},
  journal={arXiv preprint arXiv:2412.15204},
  year={2024}
}
```

### Groups, Tags, and Tasks

#### Groups

* `longbench2_single`: Single-document QA tasks requiring comprehension of documents across various domains (government, legal, literature, finance, academic, detective stories, and order of events)
* `longbench2_multi`: Multi-document QA tasks requiring information synthesis and reasoning across multiple documents in government, academic, finance, and news
* `longbench2_incontext`: Long in-context learning tasks including user guide comprehension, translation with examples, and many-shot learning scenarios
* `longbench2_history`: Long-dialogue history understanding tasks involving agent conversations and dialogue history comprehension
* `longbench2_structured`: Long structured data understanding tasks for graph and table data processing

#### Tags

* `longbench2`: Run the full benchmark with 503 multiple-choice questions (8k-2M words) testing understanding and reasoning on long-context tasks

#### Tasks

**Single-Document QA:**
* `longbench2_govt_single`: Question answering from single government documents
* `longbench2_legal_single`: Question answering from single legal documents
* `longbench2_lit_single`: Question answering from single literature/literary documents
* `longbench2_fin_single`: Question answering from single financial documents
* `longbench2_academic_single`: Question answering from single academic papers and research documents
* `longbench2_detective`: Question answering from detective stories requiring logical reasoning
* `longbench2_event_order`: Temporal reasoning tasks about event ordering in narratives

**Multi-Document QA:**
* `longbench2_govt_multi`: Question answering across multiple government documents
* `longbench2_academic_multi`: Question answering across multiple academic papers
* `longbench2_fin_multi`: Question answering across multiple financial documents
* `longbench2_news_multi`: Question answering across multiple news articles

**Long In-context Learning:**
* `longbench2_user_guide`: Comprehension and application of user guide instructions
* `longbench2_translate`: Translation tasks in new languages with long examples
* `longbench2_many_shot`: Few-shot learning with many examples in context

**Long-dialogue History Understanding:**
* `longbench2_agent_history`: Understanding and reasoning over extended agent conversation histories
* `longbench2_dialogue_history`: Understanding and reasoning over long dialogue exchanges

**Code Repository Understanding:**
* `longbench2_code`: Question answering on code repositories requiring codebase comprehension

**Long Structured Data Understanding:**
* `longbench2_graph`: Understanding and reasoning over graph-structured data
* `longbench2_table`: Understanding and reasoning over tabular data

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
