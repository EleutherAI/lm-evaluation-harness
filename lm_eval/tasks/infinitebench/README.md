# InfiniteBench

### Paper

Title: `∞Bench: Extending Long Context Evaluation Beyond 100K Tokens`

Abstract: https://arxiv.org/abs/2402.13718

InfiniteBench is the first LLM benchmark featuring an average data length surpassing 100K tokens. It includes 12 tasks spanning 5 domains (retrieval, code, math, novels, dialogue) across English and Chinese.

Homepage: https://github.com/OpenBMB/InfiniteBench

### Citation

```bibtex
@article{zhang2024infinitebench,
  title={$\infty$Bench: Extending Long Context Evaluation Beyond 100K Tokens},
  author={Zhang, Xinrong and Chen, Yingfa and Hu, Shengding and Xu, Zihang and Chen, Junhao and Hao, Moo and Han, Xu and Thai, Zhen Leng and Wang, Shuo and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2402.13718},
  year={2024}
}
```

### Groups and Tasks

#### Groups

- `infinitebench`: All 11 InfiniteBench tasks (math_calc excluded due to evaluation complexity)

#### Tasks

| Task | Split | Domain | Description | Metric |
|------|-------|--------|-------------|--------|
| `infinitebench_passkey` | passkey | Retrieval | Find a hidden passkey in irrelevant text | First-int match |
| `infinitebench_kv_retrieval` | kv_retrieval | Retrieval | Extract value for a given key from JSON | Word match |
| `infinitebench_number_string` | number_string | Retrieval | Find a hidden number in text | First-int match |
| `infinitebench_code_run` | code_run | Code | Determine output of Python function chain | Last-word match |
| `infinitebench_code_debug` | code_debug | Code | Identify which function has a bug | Last-letter match |
| `infinitebench_math_find` | math_find | Math | Find specific information in math text | First-number match |
| `infinitebench_longdialogue_qa_en` | longdialogue_qa_en | Dialogue | Answer questions about a long dialogue | Substring match |
| `infinitebench_longbook_qa_en` | longbook_qa_en | Novel (EN) | Answer questions about a book | Token F1 |
| `infinitebench_longbook_sum_en` | longbook_sum_en | Novel (EN) | Summarize a book | ROUGE-Lsum |
| `infinitebench_longbook_choice_en` | longbook_choice_en | Novel (EN) | Multiple-choice about a book | Last-letter match |
| `infinitebench_longbook_qa_chn` | longbook_qa_chn | Novel (ZH) | Answer questions about a book (Chinese) | Character F1 |

### Usage

```bash
# Run all InfiniteBench tasks
lm_eval --model hf --model_args pretrained=<model_name> --tasks infinitebench

# Run a specific task
lm_eval --model hf --model_args pretrained=<model_name> --tasks infinitebench_passkey

# With vLLM for faster inference on long contexts
lm_eval --model vllm \
  --model_args pretrained=<model_name>,max_model_len=131072 \
  --tasks infinitebench
```

### Notes

- **Context length**: Most tasks have contexts exceeding 100K tokens. Ensure your model supports sufficient context length.
- **math_calc** is excluded from this implementation as it requires complex multi-step calculation verification.
- **Dataset**: Loaded from `xinrongzhang2022/InfiniteBench` on HuggingFace Hub.
- **Evaluation methods** match the official InfiniteBench implementation. Each task uses the scoring method from the original paper: first-int extraction for retrieval, token-level F1 for QA, ROUGE-Lsum for summarization, and last-letter extraction for multiple-choice.
