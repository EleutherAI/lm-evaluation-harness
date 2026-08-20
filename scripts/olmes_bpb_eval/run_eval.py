"""Run one model/task shard and write an atomic lm-eval result JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

from lm_eval import simple_evaluate
from lm_eval.utils import handle_non_serializable


ROOT = Path(__file__).resolve().parents[2]
PINNED_PACKAGES = (
    "lm_eval",
    "transformers",
    "torch",
    "datasets",
    "accelerate",
    "antlr4-python3-runtime",
    "math-verify",
    "sympy",
    "tokenizers",
    "vllm",
    "ray",
    "numpy",
)
GIT = shutil.which("git")


def _load_hf_token() -> None:
    """Load a mounted token without printing it or copying it to results."""
    if os.environ.get("HF_TOKEN"):
        return
    token_path = os.environ.get("HF_TOKEN_PATH")
    if token_path and Path(token_path).is_file():
        os.environ["HF_TOKEN"] = Path(token_path).read_text(encoding="utf-8").strip()


def _git(args: list[str]) -> str | None:
    if GIT is None:
        return None
    try:
        return subprocess.check_output(  # noqa: S603 - fixed git executable
            [GIT, "-c", f"safe.directory={ROOT}", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_snapshot_sha256() -> str | None:
    if GIT is None:
        return None
    try:
        output = subprocess.check_output(  # noqa: S603 - fixed git executable
            [
                GIT,
                "-c",
                f"safe.directory={ROOT}",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            cwd=ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    digest = hashlib.sha256()
    for raw_path in sorted(path for path in output.split(b"\0") if path):
        path = Path(os.fsdecode(raw_path))
        if not path.is_file():
            continue
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_manifest() -> dict:
    versions = {}
    for package in PINNED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "source_snapshot_sha256": _source_snapshot_sha256(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compute-bpb", action="store_true")
    parser.add_argument("--confirm-run-unsafe-code", action="store_true")
    parser.add_argument("--limit", type=float)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--bootstrap-iters", type=int, default=0)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_hf_token()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_args = {
        "pretrained": args.model,
        "revision": args.revision,
        "tokenizer_revision": args.revision,
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "gpu_memory_utilization": 0.75,
        "max_model_len": args.max_model_len,
        "max_num_seqs": 256,
        # The cluster image's CUDA headers cannot build FlashInfer's TRT-LLM
        # decode JIT, while FlashAttention v4 on sm100 rejects Qwen3.5's
        # head_dim=256 sequence-used tensors. Triton's backend supports this
        # exact text-only BF16 configuration without either JIT path.
        "attention_backend": "TRITON_ATTN",
        # Qwen3.5 base checkpoints share a multimodal architecture, but this
        # campaign is text-only. Avoid loading/profiling the unused vision
        # tower while preserving the exact language-model weights/tokenizer.
        "language_model_only": True,
        "seed": 0,
        "data_parallel_size": args.data_parallel_size,
    }
    results = simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=[task for task in args.tasks.split(",") if task],
        batch_size=args.batch_size,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        compute_bpb=args.compute_bpb,
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
        log_samples=False,
    )
    if results is None:
        raise RuntimeError("lm-eval returned no rank-zero results")
    results["olmes_bpb_runtime"] = runtime_manifest()
    results["olmes_bpb_shard"] = {
        "tasks": args.tasks.split(","),
        "compute_bpb": args.compute_bpb,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
            ensure_ascii=False,
            default=handle_non_serializable,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
