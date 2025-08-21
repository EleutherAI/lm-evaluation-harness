# InfiniteBench

### Paper

Title: `InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens`

Abstract: `InfiniteBench is a comprehensive benchmark designed to evaluate language models on their ability to understand and process extremely long contexts (100K+ tokens). It includes 12 tasks spanning various domains including retrieval, mathematics, code understanding, question answering, summarization, and dialogue understanding. The benchmark focuses on real-world and synthetic tasks that require processing information across the entire context window.`

Homepage: `https://github.com/OpenBMB/InfiniteBench`

### Citation

```
@article{zhang2024infinitebench,
    title={InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens},
    author={Zhang, Xinrong and Chen, Yingfa and Hu, Shengding and Xu, Zihang and Chen, Junhao and Hao, Moo and Han, Xu and Thai, Zhen and Wang, Shuo and Liu, Zhiyuan and Sun, Maosong},
    journal={arXiv preprint arXiv:2402.13718},
    year={2024}
}
```

### Groups and Tasks

#### Groups

* `infinitebench`: All InfiniteBench tasks
* `infinitebench_retrieval`: Retrieval-based tasks
* `infinitebench_math`: Mathematical reasoning tasks
* `infinitebench_code`: Code understanding tasks
* `infinitebench_qa`: Question answering tasks
* `infinitebench_dialogue`: Dialogue understanding tasks

#### Tasks

The benchmark includes 12 tasks with contexts ranging from 100K to 1M+ tokens:

**Retrieval Tasks:**
* `infinitebench_passkey`: Passkey retrieval from long context (100K-1M tokens)
* `infinitebench_number_string`: Number string retrieval (100K-1M tokens)
* `infinitebench_kv_retrieval`: Key-value pair retrieval (100K-500K tokens)

**Math Tasks:**
* `infinitebench_math_find`: Finding mathematical expressions (100K-500K tokens)
* `infinitebench_math_calc`: Mathematical calculation in context (100K-500K tokens)

**Code Tasks:**
* `infinitebench_code_run`: Code execution prediction (100K-200K tokens)
* `infinitebench_code_debug`: Code debugging (100K-200K tokens)

**Question Answering Tasks:**
* `infinitebench_longbook_qa_eng`: English book QA (100K-200K tokens)
* `infinitebench_longbook_qa_chn`: Chinese book QA (100K-200K tokens)
* `infinitebench_longbook_sum_eng`: English book summarization (100K-200K tokens)

**Dialogue Tasks:**
* `infinitebench_longbook_choice_eng`: Multiple choice QA on books (100K-200K tokens)
* `infinitebench_longdialogue_qa_eng`: Long dialogue understanding (100K-200K tokens)

### Features

- **Extreme Length**: All tasks require processing 100K+ tokens
- **Diverse Domains**: Covers retrieval, math, code, QA, and dialogue
- **Real & Synthetic**: Mix of real-world and synthetic tasks
- **Multilingual**: Includes both English and Chinese tasks
- **Comprehensive Evaluation**: Tests various aspects of long-context understanding

### Evaluation Metrics

Different tasks use different evaluation metrics:
- **Retrieval tasks**: Exact match accuracy
- **Math tasks**: Numerical accuracy with tolerance
- **Code tasks**: Execution accuracy / Bug detection F1
- **QA tasks**: F1 score / ROUGE-L
- **Dialogue tasks**: Accuracy / F1 score

### Dataset Structure

Each task includes:
- `context`: The long input text (100K+ tokens)
- `question` or `prompt`: The specific query or task
- `answer` or `target`: Ground truth answer
- `task_type`: Type of task (retrieval, qa, math, etc.)
- `context_length`: Actual length of the context in tokens

### Notes

- InfiniteBench is designed to push the boundaries of current LLMs
- Tasks require not just retrieval but also reasoning over long contexts
- Performance typically degrades significantly as context length increases
- Useful for evaluating and comparing long-context capabilities across models