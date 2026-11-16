### Evaluation example

Here, we use the `--predict_only` argument and compute the performance metrics as described below.

**Step 1: Generate the predictions**

```bash
lm_eval \
  --model hf \
  --model_args pretrained=AI-Sweden-Models/Llama-3-8B \
  --tasks ask_gec \
  --output results/ask_gec/0-shot/ \
  --log_samples \
  --show_config \
  --write_out \
  --predict_only \
  --batch_size auto \
  --num_fewshot 0
```

**Step 2: Evaluate the predictions with ERRANT**

* Please refer to the installation instructions [here](https://github.com/chrisjbryant/errant/tree/main).
* Run the following:
    ```bash
    python3 ask_gec/errant.py --fpath results/ask_gec/0-shot/AI-Sweden-Models__Llama-3-8B/samples_ask_gec_p0_2025-01-28T01-08-13.454441.jsonl --out_fdir results/ask_gec/0-shot/AI-Sweden-Models__Llama-3-8B/
    ```
* The results will be saved as `results/ask_gec/0-shot/AI-Sweden-Models__Llama-3-8B/samples_ask_gec_p0_2025-01-28T01-08-13.454441_errant.json`
