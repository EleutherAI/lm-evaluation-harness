# Requires:
#   HARNESS_DIR: base directory of evaluation harness
#   TP_DEGREE: Tensor parallel degree. Communication overhead is high, so only set >1 when necessary.

PROFILE=open-web-math
ENDPOINT=codellama_7b_200btok_step42000

OUT=${HARNESS_DIR}/output/${ENDPOINT}_majk.json

source ${HARNESS_DIR}/eval_scripts/generic_run_majk.sh

