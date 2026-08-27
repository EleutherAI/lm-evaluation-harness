# MMMLU (Multilingual MMLU)

### Dataset

* Source: [openai/MMMLU](https://huggingface.co/datasets/openai/MMMLU)
* Paper: [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
* Description: Human translated versions of the 57 MMLU subjects covering 14 locales (AR_XY, BN_BD, DE_DE, ES_LA, FR_FR, HI_IN, ID_ID, IT_IT, JA_JP, KO_KR, PT_BR, SW_KE, YO_NG, ZH_CN). Each CSV exposes the prompt in a `Question` column, discrete answers in `A`/`B`/`C`/`D`, the key under `Answer`, and the subject label in `Subject`.

### Groups

* `mmmlu`: Aggregates every locale specific benchmark.
* `mmmlu_{locale}`: Locale level groups (for example `mmmlu_ar_xy`) aggregating the subject category scores for that language.
* `mmmlu_{locale}_{category}`: Category groupings (`stem`, `other`, `social_sciences`, `humanities`) scoped to one locale and implemented via tags in the YAML files.

### Notes

* Few-shot examples are drawn from the test split because the dataset only exposes test CSVs.
* `doc_to_text` mirrors the original MMLU formatting but pulls choice strings directly from the `A`/`B`/`C`/`D` columns required by the HF dataset schema.
