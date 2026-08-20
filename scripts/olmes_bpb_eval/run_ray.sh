#!/usr/bin/env bash
set -euo pipefail
export UV_COMPILE_BYTECODE=0

if [[ $# -lt 2 ]]; then
  echo "usage: run_ray.sh PHASE RESULTS_DIR [ray_driver arguments...]" >&2
  exit 2
fi

phase=$1
results_dir=$2
shift 2

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
eval_venv="${OLMES_BPB_VENV:-${repo_root}/../.lm-eval-qwen35-ray-venv}"
worker_venv="${OLMES_BPB_WORKER_VENV:-${repo_root}/../.lm-eval-qwen35-worker-venv}"
constraints="${repo_root}/scripts/olmes_bpb_eval/constraints.txt"
base_python="${OLMES_BPB_PYTHON:-/opt/venv/bin/python3}"

if [[ ! -x "${eval_venv}/bin/python" ]]; then
  uv venv --clear --python "${base_python}" "${eval_venv}"
fi
uv pip install \
  --python "${eval_venv}/bin/python" \
  --constraint "${constraints}" \
  --editable "${repo_root}[vllm,math]" \
  openpyxl==3.1.5 \
  ray==2.55.1 \
  setuptools==80.10.2

"${eval_venv}/bin/python" \
  "${repo_root}/scripts/olmes_bpb_eval/bootstrap_worker_venv.py" \
  --venv "${worker_venv}" \
  --constraints "${constraints}" \
  --repo "${repo_root}"

exec "${eval_venv}/bin/python" \
  "${repo_root}/scripts/olmes_bpb_eval/ray_driver.py" \
  --phase "${phase}" \
  --results-dir "${results_dir}" \
  --worker-python "${worker_venv}/bin/python" \
  "$@"
