#!/usr/bin/env python3
"""Convenience wrapper for running generic BOUQuET lm-eval tasks.

Examples
--------
python run_bouquet_doc.py hin_Deva-spa_Latn --model hf --model_args pretrained=...
python run_bouquet_doc.py eng_Latn-fra_Latn --model vllm --model_args pretrained=...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

GENERIC_TASK_NAME = "bouquet-doc-generic"


def parse_pair(pair: str) -> tuple[str, str]:
    """Parse an official BOUQuET language pair like 'hin_Deva-spa_Latn'."""
    if "-" not in pair:
        raise ValueError(f"Pair must look like 'src_lang-tgt_lang', got '{pair}'.")

    src_lang, tgt_lang = pair.split("-", maxsplit=1)
    src_lang = src_lang.strip()
    tgt_lang = tgt_lang.strip()

    if not src_lang or not tgt_lang:
        raise ValueError(f"Pair must look like 'src_lang-tgt_lang', got '{pair}'.")

    if src_lang == tgt_lang:
        raise ValueError("Source and target languages must differ.")

    return src_lang, tgt_lang


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a generic BOUQuET lm-eval task from an official language pair."
    )
    parser.add_argument(
        "pair",
        help="Official BOUQuET language pair like hin_Deva-spa_Latn",
    )
    parser.add_argument(
        "lm_eval_args",
        nargs=argparse.REMAINDER,
        help="Remaining arguments forwarded to lm_eval",
    )
    args = parser.parse_args()

    src_lang, tgt_lang = parse_pair(args.pair)

    metadata = {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
    }

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--tasks",
        GENERIC_TASK_NAME,
        "--metadata",
        json.dumps(metadata, separators=(",", ":")),
        *args.lm_eval_args,
    ]

    print("Running:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()