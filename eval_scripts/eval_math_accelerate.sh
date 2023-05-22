export HF_DATASETS_CACHE="/net/nfs/mosaic/seanw/.cache"
export TRANSFORMERS_CACHE="/net/nfs/mosaic/seanw/.cache"

BASE_DIR="./"
OUTPUT_DIR="output/math_algebra_easy"
mkdir -p ${OUTPUT_DIR}

MODEL_PREFIX="EleutherAI/"
MODEL="pythia-1.4b-deduped"

FEWSHOT=5
BATCH_SIZE=1

TASKS="math_algebra_easy"

python main.py --description_dict_path ${BASE_DIR}/configs/config.json \
	--model_args pretrained=${MODEL_PREFIX}${MODEL} \
	--num_fewshot ${FEWSHOT} \
	--model hf-causal \
	--use_accelerate \
	--accelerate_dtype float32 \
	--tasks ${TASKS} \
	--batch_size ${BATCH_SIZE} \
	--output_path ${OUTPUT_DIR}/${MODEL}.json
