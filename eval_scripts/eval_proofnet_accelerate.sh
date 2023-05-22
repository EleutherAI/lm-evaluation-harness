export HF_DATASETS_CACHE="/net/nfs/mosaic/seanw/.cache"
export TRANSFORMERS_CACHE="/net/nfs/mosaic/seanw/.cache"


BASE_DIR="./"
OUTPUT_DIR="./output/proofnet"
mkdir -p ${OUTPUT_DIR}

MODEL="EleutherAI/pythia-1.4b-deduped"
NAME="pythia-1.4b-deduped"

FEWSHOT=6
BATCHSIZE=8

TASKS="proofnet_autoformalize_statements,proofnet_informalize_statements"

python ${BASE_DIR}/main.py --description_dict_path ${BASE_DIR}/configs/config_proofnet.json \
	--model_args pretrained=${MODEL} \
	--num_fewshot ${FEWSHOT} \
	--model hf-causal \
	--use_accelerate \
	--accelerate_dtype float32 \
	--tasks ${TASKS} \
	--batch_size ${BATCHSIZE} \
	--output_path ${OUTPUT_DIR}/${NAME}.json  >& ${OUTPUT_DIR}/${NAME}.out &

tail -f ${OUTPUT_DIR}/${NAME}.out
#
