# Requires
#   HARNESS_DIR: base directory of evaluation harness
#   TP_DEGREE: number of available GPUs
#   PROFILE: HF model profile
#   ENDPOINT: HF model name 
#   OUT: directory of output json

MODEL=${PROFILE}/${ENDPOINT}
CONFIG=${HARNESS_DIR}/configs/majk.json

SYMBOLIC=minerva_math_prealgebra,minerva_math_algebra,minerva_math_intermediate_algebra,minerva_math_counting_and_prob,minerva_math_geometry,minerva_math_precalc,minerva_math_num_theory,ocw_courses,gsm8k
MUL_CHOICE=minerva-hendrycksTest-abstract_algebra,minerva-hendrycksTest-astronomy,minerva-hendrycksTest-college_biology,minerva-hendrycksTest-college_chemistry,minerva-hendrycksTest-college_computer_science,minerva-hendrycksTest-college_mathematics,minerva-hendrycksTest-college_physics,minerva-hendrycksTest-computer_security,minerva-hendrycksTest-conceptual_physics,minerva-hendrycksTest-electrical_engineering,minerva-hendrycksTest-elementary_mathematics,minerva-hendrycksTest-high_school_biology,minerva-hendrycksTest-high_school_chemistry,minerva-hendrycksTest-high_school_computer_science,minerva-hendrycksTest-high_school_mathematics,minerva-hendrycksTest-high_school_physics,minerva-hendrycksTest-high_school_statistics,minerva-hendrycksTest-machine_learning,math_sat_cot
TOOLS=sympy_math_prealgebra,sympy_math_algebra,sympy_math_intermediate_algebra,sympy_math_counting_and_prob,sympy_math_geometry,sympy_math_precalc,sympy_math_num_theory


cd ${HARNESS_DIR}

mkdir -p ${HARNESS_DIR}/output

# Currently, can't get MUL_CHOICE working with other kinds of tasks in the same harness call.
# When this is fixed, add TOOLS to --tasks.
python main.py --no_cache --model vllm --model_args pretrained=${MODEL} --tasks ${SYMBOLIC},${TOOLS} --output_path ${OUT} --tp_degree ${TP_DEGREE} --description_dict_path $CONFIG --limit 2
