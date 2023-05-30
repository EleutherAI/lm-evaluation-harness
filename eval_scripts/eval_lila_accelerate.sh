export HF_DATASETS_CACHE="/net/nfs/mosaic/seanw/.cache"
export TRANSFORMERS_CACHE="/net/nfs/mosaic/seanw/.cache"

MODEL="EleutherAI/pythia-1.4b-deduped"
NAME="pythia-1.4b-deduped"

BASE_DIR="./"
OUTPUT_DIR="./output/lila"
mkdir -p ${OUTPUT_DIR}

FEWSHOT=5
BATCH_SIZE=1

TASKS="lila_addsub"
#TASKS="lila_addsub,lila_multiarith,lila_GSM8k_structured,lila_deepmind_mathematics_algebra,lila_svamp_structured,lila_MATH_algebra_crowdsourced"

python ${BASE_DIR}/main.py --model_args pretrained=${MODEL} \
	--description_dict_path ${BASE_DIR}/configs/config_lila.json \
	--num_fewshot ${FEWSHOT} \
	--model hf-causal \
	--use_accelerate \
	--accelerate_dtype float32 \
	--tasks ${TASKS} \
	--output_path ${OUTPUT_DIR}/${NAME}.json \
	--batch_size ${BATCH_SIZE} 

