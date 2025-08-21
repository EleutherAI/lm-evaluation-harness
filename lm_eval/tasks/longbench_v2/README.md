# LongBench v2

### Paper

Title: `LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks`

Abstract: `LongBench v2 is a comprehensive benchmark for evaluating large language models' capabilities in handling long-context tasks. Building upon LongBench v1, this second generation benchmark introduces more challenging tasks with context lengths ranging from 8k to 2M tokens, focusing on deep reasoning and real-world applications. The benchmark includes 20 tasks across 6 categories, with particular emphasis on tasks requiring multi-hop reasoning, complex information synthesis, and long-range dependency understanding.`

Homepage: `https://github.com/THUDM/LongBench`

### Citation

```
@article{bai2024longbenchv2,
    title={LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks},
    author={Bai, Yushi and Lv, Xin and Zhang, Jiajie and others},
    journal={arXiv preprint},
    year={2024}
}
```

### Groups and Tasks

#### Groups

* `longbench_v2`: All tasks in LongBench v2
* `longbench_v2_single_doc`: Single-document understanding tasks
* `longbench_v2_multi_doc`: Multi-document synthesis tasks
* `longbench_v2_code`: Code understanding and generation tasks
* `longbench_v2_reasoning`: Complex reasoning tasks

#### Tasks

The benchmark includes 20 tasks with varying context lengths (8k-2M tokens):

**Single-Document QA:**
* `longbench_v2_narrativeqa`: Question answering on narratives
* `longbench_v2_qasper`: Scientific paper QA
* `longbench_v2_multifieldqa`: Multi-field QA

**Multi-Document QA:**
* `longbench_v2_hotpotqa`: Multi-hop reasoning
* `longbench_v2_2wikimqa`: Multi-document QA
* `longbench_v2_musique`: Multi-step reasoning

**Summarization:**
* `longbench_v2_gov_report`: Government report summarization
* `longbench_v2_multi_news`: Multi-news summarization
* `longbench_v2_book_sum`: Book summarization

**Few-shot Learning:**
* `longbench_v2_trec`: Question classification
* `longbench_v2_triviaqa`: Trivia QA
* `longbench_v2_samsum`: Dialogue summarization

**Synthetic Tasks:**
* `longbench_v2_passage_retrieval`: Passage retrieval
* `longbench_v2_passage_count`: Passage counting
* `longbench_v2_kv_retrieval`: Key-value retrieval

**Code Tasks:**
* `longbench_v2_lcc`: Long code completion
* `longbench_v2_repobench`: Repository-level code understanding
* `longbench_v2_code_debug`: Long code debugging

**Extended Context Tasks (128k-2M):**
* `longbench_v2_book_qa_eng`: English book QA
* `longbench_v2_paper_assistant`: Academic paper assistant

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?

### Notes

- LongBench v2 extends context lengths significantly compared to v1 (up to 2M tokens)
- Tasks are designed to require genuine long-context understanding, not just retrieval
- Includes both English and multilingual tasks
- Evaluation metrics vary by task type (F1, ROUGE, accuracy, etc.)