# Evaluation integrations

These integrations make ThinkRetrieve usable as a baseline in two established
research ecosystems without adding either ecosystem as a package dependency.

## lm-evaluation-harness

Install `lm-eval` and `thinkretrieve[faiss]`, build a bank with
`FaissRetriever.save`, then expose this directory to the harness:

```sh
python -m lm_eval \
  --model thinkretrieve \
  --model_args model=qwen3:4b,base_url=http://localhost:11434/v1,bank_path=/path/to/gsm8k_bank,think_budget=2048 \
  --tasks gsm8k \
  --include_path integrations/lm_eval
```

This is generation-only. It is appropriate for GSM8K-style tasks; likelihood
tasks should fail clearly rather than report misleading scores. The adapter
uses the task context as the question and lets ThinkRetrieve manage its own
intermediate probe, retrieval, injection, and final answer.

## FlashRAG

`integrations/flashrag/thinkretrieve_pipeline.py` adapts a FlashRAG-style
dataset (`question` plus `update_output`) while retaining ThinkRetrieve's
worked-example bank and mid-trace injection. This is a small adapter intended
for a FlashRAG experiment configuration, not a replacement for FlashRAG's
document-retrieval pipelines.

```python
from integrations.flashrag.thinkretrieve_pipeline import ThinkRetrievePipeline

pipeline = ThinkRetrievePipeline(
    bank_path="gsm8k_bank",
    model="qwen3:4b",
    base_url="http://localhost:11434/v1",
)
dataset = pipeline.run(dataset)
```

Keep the paper citation in any result table or configuration README:

```bibtex
@article{thinkretrieve2026,
  title = {ThinkRetrieve: Retrieval-Augmented Reasoning Traces for Test-Time Scaling},
  author = {Singh, Vaibhav and Ghosal, Soumya Suvra and Gharat, Sarvesh and Pal, Soumyabrata and Narayanam, Ramasuri and Manocha, Dinesh},
  journal = {arXiv preprint arXiv:2608.10928},
  year = {2026},
  doi = {10.48550/arXiv.2608.10928}
}
```

These adapters are proposed upstream contributions. They should be benchmarked
against the host project's conventions before opening a pull request.

## Quick test

From the ThinkRetrieve repository:

```sh
PYTHONPATH=thinkretrieve/src python -m pytest -q thinkretrieve/tests integrations/tests
```

This runs the library tests and the adapter tests without a GPU, API key, or
FlashRAG/lm-eval installation. For a real smoke run, install the optional
dependencies, build or download a saved FAISS bank, start Ollama, and run the
lm-eval command above on two GSM8K examples first. Use a fresh output directory
for each configuration.

## Add it to lm-evaluation-harness

1. Fork `EleutherAI/lm-evaluation-harness` and create a branch.
2. Copy `integrations/lm_eval/thinkretrieve_lm.py` into your fork, or install
   this repository and use `--include_path integrations/lm_eval`.
3. Install `lm-eval` and `thinkretrieve[faiss]`.
4. Run a two-example GSM8K smoke test, then a documented reproducible run.
5. Add a short README section and tests, run the harness checks, and open a PR
   explaining that this is a generation-only plugin.

The harness documents this plugin approach in its [model guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md).

## Add it to FlashRAG

1. Fork `RUC-NLPIR/FlashRAG` and create a branch.
2. Copy `integrations/flashrag/thinkretrieve_pipeline.py` into FlashRAG's
   pipeline package and import it from that package's initializer if required.
3. Add a config using a saved ThinkRetrieve example bank and an OpenAI-compatible
   model endpoint.
4. Run the host project's smallest reasoning benchmark and compare plain RAG,
   extra thinking, and ThinkRetrieve with the same model and budget.
5. Record the bank, model, seed, token accounting, latency, and failure cases;
   add tests and open a PR with the paper citation.

FlashRAG's pipeline implementation is the reference for its dataset and output
conventions. Do not claim a benchmark improvement until this host-native run is
complete.

## ThinkBooster alternative

For a test-time-scaling-focused contribution, fork `IINemo/thinkbooster`, add a
`StrategyThinkRetrieve` class under `thinkbooster/strategies/`, register it in
`tests/strategy_registry.py`, add the required unit/integration tests, and add a
Hydra strategy config. Its [strategy guide](https://github.com/IINemo/thinkbooster/blob/main/docs/core/strategy_registration.md)
lists the required files and checks.
