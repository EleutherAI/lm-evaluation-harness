# Requires
#   HARNESS_DIR: base directory of evaluation harness
#   TP_DEGREE: number of available GPUs
#   PROFILE: HF model profile
#   ENDPOINT: HF model name 
#   OUT: directory of output json
#
MODEL=${PROFILE}/${ENDPOINT}

SYMBOLIC=minerva_math*
MUL_CHOICE=minerva-hendrycksTest*
TOOLS=sympy_math*


cd ${HARNESS_DIR}

mkdir -p ${HARNESS_DIR}/output

# Currently, running --tasks set to SYMBOLIC, MUL_CHOICE, or TOOLS works, 
# But trying to run any two at once causes an error when processing prompts

python main.py --no_cache --model vllm --model_args pretrained=${MODEL} --tasks ${SYMBOLIC},${TOOLS} --output_path ${OUT} --tp_degree ${TP_DEGREE} --limit 2
