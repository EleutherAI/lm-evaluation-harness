# MRCR

This task family integrates the `openai/mrcr` benchmark into `lm-evaluation-harness`.

## Available tasks

- `mrcr_2needle`
- `mrcr_4needle`
- `mrcr_8needle`
- `mrcr` (group over `2/4/8` needles)

## Notes

- The task downloads the parquet files from `openai/mrcr` and reproduces the standalone script's scoring logic.
- Tokenization happens client-side only for filtering and per-example `max_tokens` budgeting. To parallelize this step, pass `num_tokenize_workers` and optionally `tokenize_chunk_size` through `--metadata`.
- Inference requests are sent as raw chat `messages` to an OpenAI-compatible **chat completions** endpoint (based on `local-chat-completions`), so the server applies the chat formatting.
- Greedy decoding is the default. Override decoding from the CLI with `--gen_kwargs`. Long MRCR prompts might trigger infinite generation with greedy decoding.
- Long MRCR requests often need a larger API client timeout. Set it in `--model_args`, for example `timeout=7200`.
- If `model=` points to a server-side or remote model name (i.e. not a path to a locally available model with tokenizer), pass `tokenizer` through `--metadata` so MRCR can load a local Hugging Face tokenizer for token counting and sequence-length bucketing.
- MRCR drops examples whose prompt plus expected answer exceed the max model context length.
- To run a prompt-length slice, pass `bin_l` and `bin_h` through `--metadata` (`bin_l` < X <= `bin_h`).
- Final results include overall per-sample average `score`, `AUC` (area under the curve, see [contextarena.ai](https://old.contextarena.ai/#:~:text=What%20are%20the%20different%20AUC%20scores%20and%20how%20are%20they%20calculated%3F)), per-prompt-length scores from `score_gt_0_le_8k` (0 < X <= 8192) through `score_gt_512k_le_1m` (524288 < X <= 1048576), plus matching bin counts from `count_gt_0_le_8k` through `count_gt_512k_le_1m`.

## Example

Recommended local chat run with locally served model with vLLM:

```bash
lm_eval \
  --model local-chat-completions \
  --model_args model=Qwen/Qwen3-30B-A3B-Instruct-2507,max_length=262144,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=8,timeout=7200 \
  --tasks mrcr_2needle \
  --metadata '{"tokenizer": "/data/models/Qwen/Qwen3-30B-A3B-Instruct-2507", "num_tokenize_workers": 8, "tokenize_chunk_size": 4}' \
  --gen_kwargs "temperature=1.0,top_p=0.95,top_k=20,presence_penalty=1.5,repetition_penalty=1.0,do_sample=true" \
  --output_path "output_dir/mrcr_results"
```
