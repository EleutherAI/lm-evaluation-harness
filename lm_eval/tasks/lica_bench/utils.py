"""Utilities for LICA-Bench tasks in lm-evaluation-harness.

LICA-Bench: graphic-design VLM evaluation across 7 domains
(layout, typography, SVG, templates, temporal, Lottie, category).

- Benchmark code: https://github.com/purvanshi/lica-bench
- Dataset: https://github.com/purvanshi/lica-dataset

Requires:
    pip install "lica-bench @ git+https://github.com/purvanshi/lica-bench.git"

Set LICA_BENCH_DATASET_ROOT to the path of the downloaded lica-benchmarks-dataset.
"""

import json
import os
from pathlib import Path

import datasets
from PIL import Image


DATASET_ROOT_ENV = "LICA_BENCH_DATASET_ROOT"


def _get_dataset_root():
    root = os.environ.get(DATASET_ROOT_ENV, "")
    if not root:
        raise RuntimeError(
            f"Set {DATASET_ROOT_ENV} to the path of the lica-benchmarks-dataset directory.\n"
            "  export LICA_BENCH_DATASET_ROOT=/path/to/lica-benchmarks-dataset"
        )
    p = Path(root).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"{DATASET_ROOT_ENV}={root} is not a valid directory")
    return str(p)


def _serialize_gt(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _build_dataset_for_task(task_id, **kwargs):
    """Build a HuggingFace DatasetDict for a single lica-bench task."""
    from design_benchmarks import BenchmarkRegistry
    from design_benchmarks.models.base import ModelInput, Modality

    dataset_root = _get_dataset_root()

    registry = BenchmarkRegistry()
    registry.discover()
    bench = registry.get(task_id)
    data_dir = bench.resolve_data_dir(dataset_root)
    samples = bench.load_data(data_dir, dataset_root=dataset_root)

    rows = {"question": [], "answer": [], "domain": [], "task_id": [], "images": []}

    for sample in samples:
        model_input = bench.build_model_input(sample, modality=Modality.TEXT_AND_IMAGE)
        if not isinstance(model_input, ModelInput):
            continue

        pil_images = []
        for img in model_input.images or []:
            if isinstance(img, (str, Path)):
                p = Path(img).expanduser().resolve()
                if p.is_file():
                    try:
                        pil_images.append(Image.open(str(p)).convert("RGB"))
                    except Exception:
                        pass

        text = model_input.text or ""
        meta = model_input.metadata or {}
        if meta:
            meta_str = json.dumps(meta, ensure_ascii=False, default=str)
            if len(meta_str) > 100_000:
                meta_str = meta_str[:100_000] + "...[truncated]"
            text = f"{text}\n\n[metadata]\n{meta_str}" if text else f"[metadata]\n{meta_str}"

        gt = _serialize_gt(sample.get("ground_truth", ""))

        rows["question"].append(text)
        rows["answer"].append(gt)
        rows["domain"].append(bench.meta.domain)
        rows["task_id"].append(task_id)
        rows["images"].append(pil_images if pil_images else [])

    features = datasets.Features(
        {
            "question": datasets.Value("string"),
            "answer": datasets.Value("string"),
            "domain": datasets.Value("string"),
            "task_id": datasets.Value("string"),
            "images": datasets.Sequence(datasets.Image()),
        }
    )

    ds = datasets.Dataset.from_dict(rows, features=features)
    return datasets.DatasetDict({"test": ds})


# ── Per-task dataset builders (referenced by YAML via custom_dataset) ──

def build_category_1(**kw): return _build_dataset_for_task("category-1", **kw)
def build_category_2(**kw): return _build_dataset_for_task("category-2", **kw)

def build_layout_1(**kw): return _build_dataset_for_task("layout-1", **kw)
def build_layout_2(**kw): return _build_dataset_for_task("layout-2", **kw)
def build_layout_3(**kw): return _build_dataset_for_task("layout-3", **kw)
def build_layout_4(**kw): return _build_dataset_for_task("layout-4", **kw)
def build_layout_5(**kw): return _build_dataset_for_task("layout-5", **kw)
def build_layout_6(**kw): return _build_dataset_for_task("layout-6", **kw)
def build_layout_7(**kw): return _build_dataset_for_task("layout-7", **kw)
def build_layout_8(**kw): return _build_dataset_for_task("layout-8", **kw)

def build_svg_1(**kw): return _build_dataset_for_task("svg-1", **kw)
def build_svg_2(**kw): return _build_dataset_for_task("svg-2", **kw)
def build_svg_3(**kw): return _build_dataset_for_task("svg-3", **kw)
def build_svg_4(**kw): return _build_dataset_for_task("svg-4", **kw)
def build_svg_5(**kw): return _build_dataset_for_task("svg-5", **kw)
def build_svg_6(**kw): return _build_dataset_for_task("svg-6", **kw)
def build_svg_7(**kw): return _build_dataset_for_task("svg-7", **kw)
def build_svg_8(**kw): return _build_dataset_for_task("svg-8", **kw)

def build_template_1(**kw): return _build_dataset_for_task("template-1", **kw)
def build_template_2(**kw): return _build_dataset_for_task("template-2", **kw)
def build_template_3(**kw): return _build_dataset_for_task("template-3", **kw)
def build_template_4(**kw): return _build_dataset_for_task("template-4", **kw)
def build_template_5(**kw): return _build_dataset_for_task("template-5", **kw)

def build_temporal_1(**kw): return _build_dataset_for_task("temporal-1", **kw)
def build_temporal_2(**kw): return _build_dataset_for_task("temporal-2", **kw)
def build_temporal_3(**kw): return _build_dataset_for_task("temporal-3", **kw)
def build_temporal_4(**kw): return _build_dataset_for_task("temporal-4", **kw)
def build_temporal_5(**kw): return _build_dataset_for_task("temporal-5", **kw)
def build_temporal_6(**kw): return _build_dataset_for_task("temporal-6", **kw)

def build_typography_1(**kw): return _build_dataset_for_task("typography-1", **kw)
def build_typography_2(**kw): return _build_dataset_for_task("typography-2", **kw)
def build_typography_3(**kw): return _build_dataset_for_task("typography-3", **kw)
def build_typography_4(**kw): return _build_dataset_for_task("typography-4", **kw)
def build_typography_5(**kw): return _build_dataset_for_task("typography-5", **kw)
def build_typography_6(**kw): return _build_dataset_for_task("typography-6", **kw)
def build_typography_7(**kw): return _build_dataset_for_task("typography-7", **kw)
def build_typography_8(**kw): return _build_dataset_for_task("typography-8", **kw)

def build_lottie_1(**kw): return _build_dataset_for_task("lottie-1", **kw)
def build_lottie_2(**kw): return _build_dataset_for_task("lottie-2", **kw)


# ── doc_to_image / doc_to_text / process_results ──

def doc_to_image(doc):
    """Extract PIL images from the document row."""
    return doc.get("images") or []


def doc_to_text(doc):
    """Extract the question/prompt from the document row."""
    return doc.get("question", "")


def process_results(doc, results):
    """Score a prediction against the ground-truth answer.

    Uses exact match (case-insensitive) with a substring fallback
    for when the model wraps the answer in extra text.
    """
    pred = str(results[0]).strip().lower()
    answer = str(doc.get("answer", "")).strip().lower()

    if pred == answer:
        return {"acc": 1.0}
    if answer and answer in pred:
        return {"acc": 1.0}
    return {"acc": 0.0}
