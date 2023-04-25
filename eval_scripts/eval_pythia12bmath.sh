export HF_DATASETS_CACHE="/net/nfs/mosaic/seanw/.cache"
export TRANSFORMERS_CACHE="/net/nfs/mosaic/seanw/.cache"

FEWSHOT=5
BASE_DIR="./"
OUTPUT_DIR="output/math_algebra_easy"
mkdir -p ${OUTPUT_DIR}

TASKS="math_algebra_easy"

python main.py --description_dict_path ${BASE_DIR}/configs/config.json --model_args pretrained=EleutherAI/pythia-12b-deduped --num_fewshot ${FEWSHOT} --model hf-causal --use_accelerate --tasks ${TASKS} --output_path ${OUTPUT_DIR}/pythia12bdeduped.json >& ${OUTPUT_DIR}/pythia12bdeduped.out &
