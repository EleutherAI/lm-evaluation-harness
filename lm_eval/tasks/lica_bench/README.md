# LICA-Bench

LICA-Bench is a structured evaluation suite for vision-language models on graphic design artifacts, comprising 39 tasks across 7 domains.

- **Benchmark code:** https://github.com/purvanshi/lica-bench
- **Dataset:** https://github.com/purvanshi/lica-dataset

## Domains

| Domain | Tasks | Description |
|--------|-------|-------------|
| Category | 2 | Design category classification |
| Layout | 8 | Spatial arrangement understanding and generation |
| SVG | 8 | SVG graphic comprehension and generation |
| Template | 5 | Design template understanding and generation |
| Temporal | 6 | Temporal/animation understanding and generation |
| Typography | 8 | Text/font understanding and generation |
| Lottie | 2 | Lottie animation generation |

## Setup

1. Install the lica-bench package:

```bash
pip install "lica-bench @ git+https://github.com/purvanshi/lica-bench.git"
```

2. Download the dataset:

```bash
python -c "from design_benchmarks.scripts.download_data import download; download()"
```

3. Set the dataset root environment variable:

```bash
export LICA_BENCH_DATASET_ROOT=/path/to/lica-benchmarks-dataset
```

## Usage

Run all LICA-Bench tasks:

```bash
lm_eval --model hf --model_args pretrained=<model> --tasks lica_bench
```

Run a specific domain:

```bash
lm_eval --model hf --model_args pretrained=<model> --tasks lica_bench_layout
```

Run a single task:

```bash
lm_eval --model hf --model_args pretrained=<model> --tasks lica_bench_category_1
```

## Task Groups

- `lica_bench` — all 39 tasks
- `lica_bench_category` — category domain (2 tasks)
- `lica_bench_layout` — layout domain (8 tasks)
- `lica_bench_svg` — SVG domain (8 tasks)
- `lica_bench_template` — template domain (5 tasks)
- `lica_bench_temporal` — temporal domain (6 tasks)
- `lica_bench_typography` — typography domain (8 tasks)
- `lica_bench_lottie` — Lottie domain (2 tasks)
