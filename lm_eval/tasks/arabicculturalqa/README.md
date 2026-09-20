# ArabicCulturalQA

### Paper

**Title:** Beyond MCQ: An Open-Ended Arabic Cultural QA Benchmark with Dialect Variants

**Abstract:** Large Language Models (LLMs) are increasingly used to answer everyday questions, yet their performance on culturally grounded and dialectal content remains limited across languages and their varieties. We propose a comprehensive method that (i) translates Modern Standard Arabic (MSA) multiple-choice questions (MCQs) into English and several Arabic dialects, (ii) converts them into open-ended questions (OEQs), (iii) benchmarks a range of zero-shot and fine-tuned LLMs under both MCQ and OEQ settings, and (iv) generates chain-of-thought (CoT) rationales to fine-tune models for step-by-step reasoning. Using this method, we extend an existing dataset in which QAs are parallelly aligned across language varieties, making it, to our knowledge, the first of its kind. A large portion of the resulting test set is further validated through targeted human annotation and native-speaker post-editing. We conduct extensive experiments with both open and closed models. Our findings show that (i) models underperform on Arabic dialects, showing persistent gaps in culturally grounded and dialect-specific knowledge; (ii) Arabic-centric models perform well on MCQs but struggle with OEQs; and (iii) CoT improves judged correctness while yielding mixed n-gram-based metrics.

**Homepage:** https://huggingface.co/datasets/QCRI/ArabicCulturalQA

**Paper page:** https://lrec.elra.info/lrec2026-main-408


### Citation

```
@inproceedings{bhatti-etal-2026-beyond,
  title = {Beyond MCQ: An Open-Ended Arabic Cultural QA Benchmark with Dialect Variants},
  author = {Bhatti, Hunzalah Hassan and Alam, Firoj},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {5215--5231},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  url = {https://lrec.elra.info/lrec2026-main-408},
  doi = {10.63317/2smjp2wega4e}
}
```


### Groups, Tags, and Tasks

#### Groups

* `arabicculturalqa`: Both MCQ and OEQ subgroups across all six varieties.
* `arabicculturalqa_mcq`: All six MCQ leaf tasks, size-weighted `acc` / `acc_norm`.
* `arabicculturalqa_oeq`: All six OEQ leaf tasks, size-weighted `bertscore_f1` / `rougeL_f1`.

#### Tasks

Multiple-choice (loglikelihood, metrics `acc` + `acc_norm`):

* `arabicculturalqa_mcq_msa`
* `arabicculturalqa_mcq_english`
* `arabicculturalqa_mcq_egyptian`
* `arabicculturalqa_mcq_gulf`
* `arabicculturalqa_mcq_levantine`
* `arabicculturalqa_mcq_maghrebi`

Open-ended (greedy generation, metrics `bertscore_f1` + `rougeL_f1`):

* `arabicculturalqa_oeq_msa`
* `arabicculturalqa_oeq_english`
* `arabicculturalqa_oeq_egyptian`
* `arabicculturalqa_oeq_gulf`
* `arabicculturalqa_oeq_levantine`
* `arabicculturalqa_oeq_maghrebi`


### Evaluation setup

**MCQ.** Prompt `{{question}}\n\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nالجواب:` for the Arabic dialects, or `...\nAnswer:` for English. Continuations scored: `" A"`, `" B"`, `" C"`, `" D"` (leading space), matching the project's reference script `evaluate_mcq.py`.

**OEQ.** Greedy `generate_until` (`until: ["\n\n", "\n"]`, `max_gen_toks: 256`). Predictions and gold answers are normalized via the paper's pipeline before scoring (Arabic: dediac, alef/hamza/teh-marbuta unification, character whitelist via `camel-tools`; English: WordNet lemma plus Porter stem via `nltk`; both as soft deps with graceful fallback). Then scored:

* **BERTScore F1**: `aubmindlab/bert-base-arabertv2` (12 layers) for the Arabic dialects, `bert-base-uncased` for English.
* **ROUGE-L F1**: same arabertv2 tokenizer for Arabic so ROUGE tokenization aligns with BERTScore; default tokenizer for English.

OEQ requires `bert-score`, `rouge-score`, and `transformers`. `camel-tools` and `nltk` are recommended for paper-matching normalization; the configs work without them via a regex / lowercase fallback.


### Usage

```bash
# A single dialect
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct \
        --tasks arabicculturalqa_mcq_egyptian --batch_size 8

# All MCQ dialects (size-weighted mean accuracy)
lm_eval --model hf --model_args pretrained=... \
        --tasks arabicculturalqa_mcq --batch_size 8

# Full suite (MCQ + OEQ)
lm_eval --model hf --model_args pretrained=... \
        --tasks arabicculturalqa --batch_size 8
```

System prompts and chat templates are intentionally not baked into the YAMLs so the suite stays model-agnostic. Pass `--apply_chat_template true` and/or `--system_instruction "..."` at runtime if your model expects them.


### Checklist

For adding novel benchmarks / datasets to the library:

- [x] Is the task an existing benchmark in the literature?
  - [x] Have you referenced the original paper that introduced the task? (LREC 2026; see Citation.)
  - [x] If yes, does the original paper provide a reference implementation? Reference scripts (`evaluate_mcq.py` for MCQ and `qa_eval.py` for OEQ) live with the dataset; the YAMLs reproduce their prompt format, scoring continuations, BERTScore / ROUGE-L model choices, and normalization pipeline.

If other tasks on this dataset are already supported:

- [x] Is the "Main" variant of this task clearly denoted? Yes, via the top-level `arabicculturalqa` group (MCQ + OEQ across all six varieties).
- [x] Have you provided a short sentence in a README on what each new variant adds / evaluates? See Tasks above.
- [x] Have you noted which, if any, published evaluation setups are matched by this variant? See Evaluation setup above.
