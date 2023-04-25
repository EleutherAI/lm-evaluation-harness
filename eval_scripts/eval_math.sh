export HF_DATASETS_CACHE="/net/nfs/mosaic/seanw/.cache"
export TRANSFORMERS_CACHE="/net/nfs/mosaic/seanw/.cache"

FEWSHOT=5
OUTPUT_DIR="output/math_algebra_easy"
mkdir -p ${OUTPUT_DIR}

TASKS="math_algebra_easy"

python main.py --description_dict_path configs/config.json --model_args pretrained=EleutherAI/pythia-1.4b-deduped --num_fewshot ${FEWSHOT} --model gpt2 --tasks ${TASKS} --device 1 --output_path ${OUTPUT_DIR}/pythia1.4bdeduped.json >& ${OUTPUT_DIR}/pythia1.4bdeduped.out &
python main.py --description_dict_path configs/config_large.json --model_args pretrained=EleutherAI/pythia-6.9b-deduped --num_fewshot ${FEWSHOT} --model gpt2 --tasks ${TASKS} --device 2 --output_path ${OUTPUT_DIR}/pythia6.9bdeduped.json >& ${OUTPUT_DIR}/pythia6.9bdeduped.out &
python main.py --description_dict_path configs/config.json --model_args pretrained=hoskinson-center/proofGPT-v0.1 --num_fewshot ${FEWSHOT} --model gpt2 --tasks ${TASKS} --device 3 --output_path ${OUTPUT_DIR}/proofGPT-v0.1.json >& ${OUTPUT_DIR}/proofGPT-v0.1.out &
python main.py --description_dict_path configs/config_large.json --model_args pretrained=hoskinson-center/proofGPT-v0.1-6.7B --num_fewshot ${FEWSHOT} --model gpt2 --tasks ${TASKS} --device 4 --output_path ${OUTPUT_DIR}/proofGPT-v0.1-6.7B.json >& ${OUTPUT_DIR}/proofGPT-v0.1-6.7B.out &
