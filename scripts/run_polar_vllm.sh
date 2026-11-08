#!/usr/bin/env bash
# Run inside an allocated GPU job, with this branch installed in its environment.
set -euo pipefail
model=${1:?Usage: bash scripts/run_polar_vllm.sh MODEL [additional lm-eval arguments]}
shift
: "${POLAR_ATTACKER_BASE_URL:?Set the fixed B endpoint for protocol 5}"
export POLAR_MODEL_A_NAME="$model"
export POLAR_THINKING_MODE=${POLAR_THINKING_MODE:-final_only}
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Only non-reasoning models are supported in this integration.
if [[ "$POLAR_THINKING_MODE" != final_only ]]; then
    echo "Only POLAR_THINKING_MODE=final_only is supported" >&2
    exit 2
fi
model_args="pretrained=$model,enable_thinking=false"
if [[ -n "${POLAR_VLLM_ARGS:-}" ]]; then
    model_args+=",$POLAR_VLLM_ARGS"
fi
exec python -m lm_eval run \
    --model vllm --model_args "$model_args" \
    --tasks polar_bench --include_path "$root/lm_eval/tasks/polar_bench" \
    --apply_chat_template --num_fewshot 0 --log_samples \
    --output_path "${POLAR_OUTPUT_PATH:-results/polar_bench}" "$@"
