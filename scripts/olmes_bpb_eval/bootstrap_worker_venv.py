"""Build the shared evaluation venv using a GPU worker's Python runtime."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    return parser.parse_args()


def install_worker_venv(venv: str, constraints: str, repo: str) -> str:
    """Install once on a GPU node so the venv embeds its Python path."""
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv is unavailable on the Ray GPU worker")

    venv_path = Path(venv)
    python = venv_path / "bin/python"
    install_env = dict(os.environ, UV_COMPILE_BYTECODE="0")
    if not python.is_file():
        subprocess.run(  # noqa: S603 - fixed uv executable
            [uv, "venv", "--clear", "--python", sys.executable, str(venv_path)],
            check=True,
            env=install_env,
        )
    subprocess.run(  # noqa: S603 - fixed uv executable
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--constraint",
            constraints,
            "setuptools==80.10.2",
        ],
        check=True,
        env=install_env,
    )
    subprocess.run(  # noqa: S603 - fixed uv executable
        [
            uv,
            "pip",
            "install",
            "--no-build-isolation",
            "--python",
            str(python),
            "--constraint",
            constraints,
            "--editable",
            f"{repo}[vllm]",
            "openpyxl==3.1.5",
            "ray==2.55.1",
            "setuptools==80.10.2",
        ],
        check=True,
        env=install_env,
    )
    subprocess.run(  # noqa: S603 - exact interpreter and fixed import probe
        [
            str(python),
            "-c",
            "import lm_eval, ray, torch, transformers, vllm",
        ],
        check=True,
    )
    return str(python)


def main() -> None:
    args = parse_args()
    import ray

    ray.init(address="auto", log_to_driver=True)
    installer = ray.remote(num_gpus=0.01, num_cpus=1)(install_worker_venv)
    worker_python = ray.get(
        installer.remote(
            str(args.venv.resolve()),
            str(args.constraints.resolve()),
            str(args.repo.resolve()),
        )
    )
    print(f"Worker evaluation interpreter ready: {worker_python}", flush=True)


if __name__ == "__main__":
    main()
