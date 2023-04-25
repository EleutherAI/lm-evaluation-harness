export HF_DATASETS_CACHE="/net/nfs/mosaic/seanw/.cache"
export TRANSFORMERS_CACHE="/net/nfs/mosaic/seanw/.cache"

FEWSHOT=6
BATCHSIZE=8

TASKS="proofnet_autoformalize_statements,proofnet_informalize_statements"
BASE_DIR="./"
OUTPUT_DIR="./output/proofnet"
mkdir -p ${OUTPUT_DIR}

MODEL=EleutherAI/pythia-12b-deduped
NAME=pythia-12b-deduped

python ${BASE_DIR}/main.py --description_dict_path ${BASE_DIR}/configs/proofnet_config.json --model_args pretrained=${MODEL} --num_fewshot ${FEWSHOT} --model hf-causal --batch_size ${BATCHSIZE} --tasks ${TASKS} --use_accelerate --output_path ${OUTPUT_DIR}/${NAME}.json  >& ${OUTPUT_DIR}/${NAME}.out &

tail -f ${OUTPUT_DIR}/${NAME}.out
#
