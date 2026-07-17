# Task-name

## Paper

Title: VERADATA: A Standardised German Visual Question Answering Benchmark from Educational Assessments

Abstract: [will be provided later]

A manually curated, mainly German multiple-choice Visual Question Answering dataset that covers diverse domains and a range of competency levels
The original data is designed by the _Institut zur Qualitätsentwicklung im Bildungswesen_ to test German school children in classes 3, 8 and 12. It is freely accessible at their [website](https://www.iqb.hu-berlin.de/de/schule/aufgaben/).

Details of the dataset can be found here: https://huggingface.co/datasets/DFKI-SLT/vera

### Citation

```text
@inproceedings{,
  author    = {},
  title     = {VERADATA: A Standardised German Visual Question Answering Benchmark from Educational Assessments},
  booktitle = {},
  year      = {2026},
  note      = {To appear}
}
```

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
