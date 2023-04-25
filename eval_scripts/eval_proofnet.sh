export HF_DATASETS_CACHE="/net/nfs/mosaic/seanw/.cache"
export TRANSFORMERS_CACHE="/net/nfs/mosaic/seanw/.cache"

FEWSHOT=6
BATCHSIZE=8

TASKS="proofnet_autoformalize_statements,proofnet_informalize_statements"
BASE_DIR="./"
OUTPUT_DIR="./output/proofnet"
mkdir -p ${OUTPUT_DIR}


python ${BASE_DIR}/main.py --description_dict_path ${BASE_DIR}/configs/proofnet_config.json --model_args pretrained=EleutherAI/pythia-6.9b-deduped --num_fewshot ${FEWSHOT} --model hf-causal --batch_size ${BATCHSIZE} --tasks ${TASKS} --device cuda:1 --output_path ${OUTPUT_DIR}/pythia6.9bdeduped.json  >& ${OUTPUT_DIR}/pythia6.9bdeduped.out &

