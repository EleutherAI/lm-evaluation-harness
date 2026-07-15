# Task-name

## Paper

Title: KANGAROOBENCH: A Native German Benchmark for Evaluating the Modality Gap in Visual Mathematics

Abstract: [will be provided later]

A German benchmark from the official [Känguru der Mathematik](https://www.mathe-kaenguru.de/) competition consisting of multi-modal multiple-choice math questions.
Details of the dataset can be found here: https://huggingface.co/datasets/DFKI-SLT/kangaroo_math_benchmark

### Citation

```text
@inproceedings{hug2026kangaroobench,
  author    = {Hug, Dennis and Wolf, Ren\'e and Thomas, Philippe},
  title     = {KangarooBench: A Native German Benchmark for Evaluating the Modality Gap in Visual Mathematics},
  booktitle = {Conference and Labs of the Evaluation Forum (CLEF 2026)},
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
    --tasks kangaroo_math \
    --device cuda:0 \
    --batch_size 2
```
