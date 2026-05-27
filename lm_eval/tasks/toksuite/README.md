
# TokSuite

## Paper
Title: `TokSuite: Measuring the Impact of Tokenizer Choice on Language Model Behavior`

Abstract: [2512.20757](https://arxiv.org/abs/2512.20757)

Tokenizers provide the fundamental basis through which text is represented and processed by language models (LMs). Despite the importance of tokenization, its role in LM performance and behavior is poorly understood due to the challenge of measuring the impact of tokenization in isolation. To address this need, we present TokSuite, a collection of models and a benchmark that supports research into tokenization's influence on LMs. Specifically, we train fourteen models that use different tokenizers but are otherwise identical using the same architecture, dataset, training budget, and initialization. Additionally, we curate and release a new benchmark that specifically measures model performance subject to real-world perturbations that are likely to influence tokenization. Together, TokSuite allows robust decoupling of the influence of a model's tokenizer, supporting a series of novel findings that elucidate the respective benefits and shortcomings of a wide range of popular tokenizers.

Homepage: [https://huggingface.co/toksuite](https://huggingface.co/toksuite)

### Citation
```
@misc{toksuite2025,
      title={TokSuite: Measuring the Impact of Tokenizer Choice on Language Model Behavior},
      author={Gül Sena Altıntaş and Malikeh Ehghaghi and Brian Lester and Fengyuan Liu and Wanru Zhao and Marco Ciccone and Colin Raffel},
      year={2025},
      eprint={2512.20757},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.20757},
}
```

## Groups and Tasks
TokSuite benchmark consists of seven individual datasets, each with simple multiple-choice completion questions:
- **Parallel Multilingual Benchmark**: There are 40 sentences (denoted by the English canonical subset, i.e. `toksuite_english_canonical`), which are then translated into canonical questions in the four target languages: Turkish (`tur_Latn`), Chinese (`zho_Hans`), Italian (`ita_Latn`), and Farsi/Persian (`pes_Arab`) by native speakers. Each dataset contains naturally possible variations of the canonical question, reflecting linguistic variaties of each language.
- **STEM Benchmark**: This benchmark contains multiple-choice completion questions spanning STEM subjects.
- **MATH Benchmark**: This benchmark contains simple arithmetic questions, also translated into the target languages

### Groups
### Tasks

- toksuite_english
    - toksuite_english_word_reordering  
    - toksuite_english_lowercase  
    - toksuite_english_canonical  
    - toksuite_english_space_removal  
    - toksuite_english_keyboard_proximity_errors  
    - toksuite_english_macron_diacritic  
    - toksuite_english_character_substitution  
    - toksuite_english_scripted_text  
    - toksuite_english_grammatical_errors  
    - toksuite_english_homoglyphs  
    - toksuite_english_letter_repetition_for_emphasis  
    - toksuite_english_spaced_styling  
    - toksuite_english_compounds  
    - toksuite_english_colloquial  
    - toksuite_english_historical_spelling  
    - toksuite_english_similar_words  
    - toksuite_english_abbreviations  
    - toksuite_english_hyphenated_spelling  
    - toksuite_english_web_search_query  
    - toksuite_english_spelled_out  
    - toksuite_english_date_formats  
    - toksuite_english_orthographic_errors  
    - toksuite_english_capitalization  
    - toksuite_english_emoji_substitution  
    - toksuite_english_contractions  
    - toksuite_english_superscript_subscript_styling  
    - toksuite_english_character_deletion  
    - toksuite_english_inflections  
    - toksuite_english_ocr_errors
- toksuite_turkish
    - toksuite_turkish_derivations  
    - toksuite_turkish_typographical_errors  
    - toksuite_turkish_date_formats  
    - toksuite_turkish_web_search_query  
    - toksuite_turkish_english_keyboard  
    - toksuite_turkish_equivalent_expressions  
    - toksuite_turkish_dialects  
    - toksuite_turkish_canonical  
    - toksuite_turkish_code_language_script_switching  
    - toksuite_turkish_colloquial  
    - toksuite_turkish_inflections  
    - toksuite_turkish_similar_words  
    - toksuite_turkish_keyboard_proximity_errors  
    - toksuite_turkish_word_reordering  
    - toksuite_turkish_spelled_out  
    - toksuite_turkish_grammatical_errors  
    - toksuite_turkish_orthographic_errors
- toksuite_chinese
    - toksuite_chinese_keyboard_proximity_errors  
    - toksuite_chinese_word_spacing_zero-width_characters_extra_space  
    - toksuite_chinese_romanization  
    - toksuite_chinese_spelled_out  
    - toksuite_chinese_partially_romanized  
    - toksuite_chinese_colloquial  
    - toksuite_chinese_canonical  
    - toksuite_chinese_equivalent_expressions  
    - toksuite_chinese_code_language_script_switching  
    - toksuite_chinese_optional_diacritics  
    - toksuite_chinese_ocr_errors  
    - toksuite_chinese_space_removal  
    - toksuite_chinese_traditional
- toksuite_italian
    - toksuite_italian_english_keyboard  
    - toksuite_italian_date_formats  
    - toksuite_italian_typographical_errors  
    - toksuite_italian_orthographic_errors  
    - toksuite_italian_abbreviations  
    - toksuite_italian_code_language_script_switching  
    - toksuite_italian_phonetic_spelling  
    - toksuite_italian_dialects  
    - toksuite_italian_similar_words  
    - toksuite_italian_grammatical_errors  
    - toksuite_italian_contractions  
    - toksuite_italian_web_search_query  
    - toksuite_italian_spelled_out  
    - toksuite_italian_keyboard_proximity_errors  
    - toksuite_italian_numerical_formats  
    - toksuite_italian_capitalization  
    - toksuite_italian_canonical  
    - toksuite_italian_plausible_diacritics_errors
- toksuite_farsi
    - toksuite_farsi_colloquial  
    - toksuite_farsi_optional_diacritics  
    - toksuite_farsi_romanization  
    - toksuite_farsi_spelled_out  
    - toksuite_farsi_code_language_script_switching  
    - toksuite_farsi_word_spacing_zero-width_characters_extra_space  
    - toksuite_farsi_arabic_keyboard_for_farsi  
    - toksuite_farsi_keyboard_proximity_errors  
    - toksuite_farsi_canonical  
    - toksuite_farsi_dialects  
    - toksuite_farsi_number_romanization  
    - toksuite_farsi_equivalent_expressions
- toksuite_stem
    - toksuite_stem_unusual_formatting  
    - toksuite_stem_unicode_formatting  
    - toksuite_stem_latex  
    - toksuite_stem_superscript_subscript  
    - toksuite_stem_character_deletion  
    - toksuite_stem_colloquial  
    - toksuite_stem_equivalent_expressions  
    - toksuite_stem_diacriticized_styling  
    - toksuite_stem_enclosed_characters  
    - toksuite_stem_upside_down_rotated  
    - toksuite_stem_canonical  
    - toksuite_stem_scripted_text  
    - toksuite_stem_strikethrough  
    - toksuite_stem_morpheme_separation  
    - toksuite_stem_spelled_out  
    - toksuite_stem_space_removal  
    - toksuite_stem_fullwidth_characters  
    - toksuite_stem_typographical_errors  
    - toksuite_stem_double_struck
- toksuite_math
    - toksuite_math_decorative_unicode  
    - toksuite_math_chinese  
    - toksuite_math_space_removal  
    - toksuite_math_latex  
    - toksuite_math_farsi  
    - toksuite_math_spelled_out  
    - toksuite_math_canonical  
    - toksuite_math_turkish  
    - toksuite_math_italian


### Task Validity Checklist

The checklist is the following:

For adding novel benchmarks/datasets to the library:

- [x] Is the task an existing benchmark in the literature?
  - [x] Have you referenced the original paper that introduced the task?
  - [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

- [ ] Is the "Main" variant of this task clearly denoted?
- [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
- [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

## Reproducing Results in the Paper
As part of TokSuite, we are interested in two metrics: 1) raw accuracy on each subtask 2) the drop in accuracy compared to the canonical form of each example.
One needs to cross-reference the results on the canonical subset and the perturbed subtask to compute, hence these metrics need to be computed manually after the evaluation is run. Below, we outline steps to reproduce the robustness metric and tables in the paper.

We are interested in the measure of robustness, which is the relative performance drop compared to the canonical (i.e. most natural form) represented as
    $$\frac {(\textrm{Acc}_\textrm{can} - \textrm{Acc}_\textrm{pert})} {\textrm{Acc}_\textrm{can}},$$
where $\textrm{Acc}_\textrm{can}$ is the mean accuracy on the corresponding canonical subset and $\textrm{Acc}_\textrm{pert}$ corresponds to the mean accuracy in the consider subtask or group. Below, we outline ways to reproduce tables from the paper.

To compute this metric, TokSuite provides additional metadata in the task configs, e.g. corresponding canonical task (named as `canonical_task`) and number of samples in this perturbed subset (named as `num_samples`) so that we can recompute the weighted average.

Another way is to log all samples and operate on the sample level however, it is easier to run benchmark all together and run the processing on the results files. Running the benchmark on a 1 to 2B parameter model does not take more than 10 minutes on an L40S gpu.

1. Run the benchmark
```bash
OUTPUT_PATH="/tmp/lm_eval/toksuite"
mkdir -p $OUTPUT_PATH
lm-eval --model hf --model_args pretrained="toksuite/gpt2,tokenizer=openai-community/gpt2" \
    --tasks toksuite --log_samples \
    --output_path=$OUTPUT_PATH
```

2. Process results, pass either "latex", "markdown", or "dataframe" to get the formatted table

```python
from pathlib import Path
results_paths = [
    list(Path("/tmp/lm_eval/toksuite/openai-community__gpt2").rglob("results*.json"))[-1],
    ### more models if you ran them
    ]
from lm_eval.tasks.toksuite.utils import LATEX_TABLE_CATEGORIES, get_table_str
tab = get_table_str(results_paths, output_format="latex")
print(tab)
```

3. Reporting accuracies is also easy
```python
from pathlib import Path
results_paths = [
    list(Path("/tmp/lm_eval/toksuite/openai-community__gpt2").rglob("results*.json"))[-1],
    ### more models if you ran them
    ]
from lm_eval.tasks.toksuite.utils import get_accuracy_table_str, get_canonical_performance_table_str, LATEX_TABLE_CATEGORIES_CANONICAL
tab = get_accuracy_table_str(results_paths)
print(tab)
## or simply the canonical accuracies
canonical_accuracies = get_canonical_performance_table_str(results_path, latex_table_categories=LATEX_TABLE_CATEGORIES_CANONICAL, latex_column_data=LATEX_TABLE_CATEGORIES_CANONICAL.keys())
print(canonical_accuracies)
```
