### IberoBench, instructed

IberoBench task variants that are adapted for instruction-tuned and/or aligned models. The changes mostly consist in different stop criteria (`generation_kwargs.until`) and generation limit (`generation_kwargs.max_gen_toks`) to avoid issues where generation would be cut in the middle due to the model's usage of `\n\n` between paragraphs, and incomplete generations due to limits that were too short for aligned models that provide CoT.

All variants in this folder use `_instruct` in their task names. These tasks are intended for instruction-tuned or aligned models. Inspect logged samples when evaluating a new model family to confirm that its chat template and answer formatting match the task.

* **`mgsm_direct_{en,es,ca}_instruct`**: MathVerify-enabled direct-answer variants. They retain the legacy numeric exact-match metric, add symbolic/numeric equivalence checking with `math_verify`, allow up to 8191 generated tokens, and use no stop strings. Catalan uses `projecte-aina/mgsm_ca`; English and Spanish use the corresponding `juletxara/mgsm` subsets.
* **`mgsm_direct_{eu,gl}_instruct`**, **`mgsm_native_cot_eu_instruct`**: Existing Basque and Galician generation-setting variants, renamed without changing their evaluation behavior.
* **`cabreu_instruct`** tag, with tasks **`cabreu_{abstractive, extractive, extreme}_instruct`**: Longer `max_gen_toks` (8191); no generation stop criteria (modified in a new `_cabreu_instruct_common_yaml`).
* **`summarization_gl_instruct`**, **`xlsum_es_instruct`**: Longer `max_gen_toks` (1024); no generation stop criteria.
* **`xsum_instruct`**: New implementation of `xsum` as a regular summarization task, without using `unitxt`. Different prompt that asks for a *"One-sentence summary"*; use the same processing function as `summarization_gl` (`process_summarization` from `galician_bench/utils.py`); longer `max_gen_toks` (1024); no stop criteria.
* **`xquad_{ca,es,en}_instruct`**: Stop only on `\n\n` and not on `\n`. (`xquad_{en,es}_instruct` import from `_xquad_instruct_common_yaml`, while `xquad_ca_instruct` is defined on its own.)


### Multiple-choice instruct variants

These variants preserve multiple-choice likelihood evaluation. They add an explicit instruction, render labeled choices, and place the candidate label immediately after an `<answer>` assistant prefix. This reduces prompt ambiguity for chat models without changing the benchmark into free-form generation.

| Family | Task IDs | Languages | Adaptation |
| --- | --- | --- | --- |
| ARC Easy/Challenge | `arc_{easy,challenge}_instruct`, `arc_ca_{easy,challenge}_instruct`, `arc_es_{easy,challenge}_instruct` | English, Catalan, Spanish | Localized question/choice labels and instructions; scores the original answer labels. |
| Belebele | `belebele_{eng,cat,spa}_Latn_instruct` | English, Catalan, Spanish | Uses a shared local template and scores A/B/C/D after `<answer>`. Passage and question content remain unchanged. |
| HellaSwag | `hellaswag_instruct` | English | Makes the four endings explicit labeled choices while retaining canonical HellaSwag preprocessing and `acc`/`acc_norm`. |
| MMLU / MMMLU | `mmlu_en_instruct`, `mmmlu_es_instruct` | English, Spanish | Uses localized instructions and prompts and scores A/B/C/D after `<answer>`. English uses the aggregate `all` configuration from `cais/mmlu`; Spanish uses the `ES_LA` translation from `openai/MMMLU`. |
| OpenBookQA | `openbookqa_instruct`, `openbookqa_ca_instruct`, `openbookqa_es_instruct` | English, Catalan, Spanish | Localized instructions and labeled choices; preserves the source splits, targets, decontamination query, and accuracy metrics. |
| Social IQA | `social_iqa_instruct`, `siqa_ca_instruct`, `siqa_es_instruct` | English, Catalan, Spanish | Localized context/question/choice prompts and constrained A/B/C scoring; preserves each dataset’s original validation split. |

### MathVerify MGSM variants

`mgsm_direct_{en,es,ca}_instruct` are maintained here. These variants keep a numeric exact-match score for comparison and add `math_verify`, which accepts mathematically equivalent numeric forms. They remove stop strings and allow up to 8191 generated tokens so instruction-tuned models can complete their reasoning. English and Spanish use `juletxara/mgsm`; Catalan uses `projecte-aina/mgsm_ca` with a localized `Pregunta`/`Resposta` prompt.
