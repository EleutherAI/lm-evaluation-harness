# SEGData

## Paper

Title: SEGDATA: A Standardised German Visual Question Answering Benchmark from Educational Assessments

Abstract: [will be provided later]

A manually curated, mainly German multiple-choice Visual Question Answering dataset that covers diverse domains and a range of competency levels.
The original data is designed by the _Institut zur Qualitätsentwicklung im Bildungswesen_ to test German school children in the VERA comparison tests for classes 3 and 8. It is freely accessible at their [website](https://www.iqb.hu-berlin.de/de/schule/aufgaben/).

Homepage: https://huggingface.co/datasets/DFKI-SLT/SEGData

### Citation

```text
@inproceedings{yuzuncuoglu2026segdata,
  author    = {Y{\"u}z{\"u}nc{\"u}oglu, Imge and Barth, Fabio and Wolf, Ren{\'e} and Thomas, Philippe},
  title     = {SEGDATA: A Standardised German Visual Question Answering Benchmark from Educational Assessments},
  year      = {2026},
  note      = {To appear}
}
```

### Groups, Tags, and Tasks

#### Groups

None.

#### Tags

None.

#### Tasks

* `vera`: Multiple-choice visual question answering over 395 items from German standardised educational assessments. The model is shown an image plus a description and question, and must reply with the letter of the correct option.

The task is named after the VERA (_Vergleichsarbeiten_) assessments the items are drawn from, while the dataset and this directory are named SEGData.

### Dataset notes

* The dataset ships a single `train` split of 395 items, which is used in full as the evaluation set. There is no held-out split, and no few-shot split — the task is intended to be run zero-shot.
* The `class` column records the school year the original assessment targets: `8` (237 items), `3` (120 items), and `k.A.` (38 items, year not recorded).
* Items carry between 2 and 16 answer options; `gold_answer` is always a single uppercase letter (`A`–`M`).
* Roughly half the items (200 of 395) were converted to multiple choice by the dataset authors; these are flagged with `added_multiplechoice: true`.
* The instruction block in the prompt is written in German for every item, including the minority of items whose description and question are in English.

### Usage

```bash
lm_eval \
    --model hf-multimodal \
    --model_args pretrained=Qwen/Qwen3-VL-32B-Instruct,dtype=bfloat16,convert_img_format=True \
    --apply_chat_template \
    --tasks vera \
    --device cuda:0 \
    --batch_size 2
```

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
