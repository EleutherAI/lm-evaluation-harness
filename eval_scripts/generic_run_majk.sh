# Requires
#   HARNESS_DIR: base directory of evaluation harness
#   TP_DEGREE: number of available GPUs
#   PROFILE: HF model profile
#   ENDPOINT: HF model name 
#   OUT: directory of output json

MODEL=${PROFILE}/${ENDPOINT}
CONFIG=${HARNESS_DIR}/configs/majk.json

# GSM8k does not support majority voting yet
SYMBOLIC=minerva_math*,ocw_courses,gsm8k
MUL_CHOICE=minerva-hendrycksTest*,math_sat_cot
TOOLS=sympy_math*
TASKS=${SYMBOLIC},${MUL_CHOICE},${TOOLS}

cd ${HARNESS_DIR}

mkdir -p ${HARNESS_DIR}/output

# Note that num_fewshot only applies to gsm8k, since the rest of the tasks have fixed prompts
python main.py --no_cache --model vllm --model_args pretrained=${MODEL} --tasks $TASKS --output_path ${OUT} --tp_degree ${TP_DEGREE} --description_dict_path $CONFIG
