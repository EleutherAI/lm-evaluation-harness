# MILU

**Original GitHub Repo:** [https://github.com/AI4Bharat/MILU](https://github.com/AI4Bharat/MILU)

### Paper

Title: `MILU: A Multi-task Indic Language Understanding Benchmark`

Abstract: `Evaluating Large Language Models (LLMs) in low-resource and linguistically diverse languages remains a significant challenge in NLP, particularly for languages using non-Latin scripts like those spoken in India. Existing benchmarks predominantly focus on English, leaving substantial gaps in assessing LLM capabilities in these languages. We introduce MILU, a Multi task Indic Language Understanding Benchmark, a comprehensive evaluation benchmark designed to address this gap. MILU spans 8 domains and 42 subjects across 11 Indic languages, reflecting both general and culturally specific knowledge. With an India-centric design, incorporates material from regional and state-level examinations, covering topics such as local history, arts, festivals, and laws, alongside standard subjects like science and mathematics. We evaluate over 42 LLMs, and find that current LLMs struggle with MILU, with GPT-4o achieving the highest average accuracy at 72 percent. Open multilingual models outperform language-specific fine-tuned models, which perform only slightly better than random baselines. Models also perform better in high resource languages as compared to low resource ones. Domain-wise analysis indicates that models perform poorly in culturally relevant areas like Arts and Humanities, Law and Governance compared to general fields like STEM. To the best of our knowledge, MILU is the first of its kind benchmark focused on Indic languages, serving as a crucial step towards comprehensive cultural evaluation. All code, benchmarks, and artifacts will be made publicly available to foster open research.`


### Citation

```bibtex
@article{verma2024milu,
  title   = {MILU: A Multi-task Indic Language Understanding Benchmark},
  author  = {Sshubam Verma and Mohammed Safi Ur Rahman Khan and Vishwajeet Kumar and Rudra Murthy and Jaydeep Sen},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2411.02538}
}
```

## Usage

##### Prerequisites

- Python 3.7+
- `lm-eval-harness` library
- HuggingFace Transformers
- vLLM (optional, for faster inference)\

1. Clone this repository:

```bash
git clone --depth 1 https://github.com/AI4Bharat/MILU.git
cd MILU
pip install -e .
```

2. Request access to the HuggingFace 🤗 dataset [here](https://huggingface.co/datasets/ai4bharat/MILU).

3. Set up your environment variables:

```bash
export HF_HOME=/path/to/HF_CACHE/if/needed
export HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```


## Supported Languages
- Bengali
- English
- Gujarati
- Hindi
- Kannada
- Malayalam
- Marathi
- Odia
- Punjabi
- Tamil
- Telugu

## HuggingFace Evaluation

For HuggingFace models, you may use the following sample command:

```bash
lm_eval --model hf \
    --model_args 'pretrained=google/gemma-2-27b-it,temperature=0.0,top_p=1.0,parallelize=True' \
    --tasks milu \
    --batch_size auto:40 \  
    --log_samples \
    --output_path $EVAL_OUTPUT_PATH \
    --max_batch_size 64 \
    --num_fewshot 5 \
    --apply_chat_template
```

## vLLM Evaluation

For vLLM-compatible models, use the following command:

```bash
lm_eval --model vllm \
    --model_args 'pretrained=meta-llama/Llama-3.2-3B,tensor_parallel_size=$N_GPUS' \
    --gen_kwargs 'temperature=0.0,top_p=1.0' \
    --tasks milu \
    --batch_size auto \
    --log_samples \
    --output_path $EVAL_OUTPUT_PATH
```

## Single Language Evaluation

To evaluate your on a specific language, modify the `--tasks` parameter:

```bash
--tasks milu_English
```

Replace `English` with the available language (e.g., Odia, Hindi, etc.).

### Evaluation Tips & Observations

1. Make sure to use `--apply_chat_template` for Instruction-fine-tuned models, to format the prompt correctly.
2. vLLM generally works better with Llama models, while Gemma models work better with HuggingFace.
3. If vLLM encounters out-of-memory errors, try reducing `max_gpu_utilization` else switch to HuggingFace.
4. For HuggingFace, use `--batch_size=auto:<n_batch_resize_tries>` to re-select the batch size multiple times.
5. When using vLLM, pass generation kwargs in the `--gen_kwargs` flag. For HuggingFace, include them in `model_args`.
