export HF_DATASETS_CACHE="/net/nfs/mosaic/seanw/.cache"
export TRANSFORMERS_CACHE="/net/nfs/mosaic/seanw/.cache"

FEWSHOT=5
OUTPUT_DIR="output/lila"
mkdir -p ${OUTPUT_DIR}

TASKS="lila_addsub,lila_multiarith,lila_GSM8k_structured,lila_deepmind_mathematics_algebra,lila_svamp_structured,lila_MATH_algebra_crowdsourced"

python main.py --model_args pretrained=EleutherAI/pythia-1.4b-deduped --num_fewshot ${FEWSHOT} --model gpt2 --tasks ${TASKS} --device 0 --output_path ${OUTPUT_DIR}/pythia1.4bdeduped.json >& ${OUTPUT_DIR}/pythia1.4bdeduped.out &
python main.py --model_args pretrained=EleutherAI/pythia-6.9b-deduped --num_fewshot ${FEWSHOT} --model gpt2 --tasks ${TASKS} --device 1 --output_path ${OUTPUT_DIR}/pythia6.9bdeduped.json >& ${OUTPUT_DIR}/pythia6.9bdeduped.out &
python main.py --model_args pretrained=hoskinson-center/proofGPT-v0.1 --num_fewshot ${FEWSHOT} --model gpt2 --tasks ${TASKS} --device 2 --output_path ${OUTPUT_DIR}/proofGPT-v0.1.json >& ${OUTPUT_DIR}/proofGPT-v0.1.out &
python main.py --model_args pretrained=hoskinson-center/proofGPT-v0.1-6.7B --num_fewshot ${FEWSHOT} --model gpt2 --tasks ${TASKS} --device 3 --output_path ${OUTPUT_DIR}/proofGPT-v0.1-6.7B.json >& ${OUTPUT_DIR}/proofGPT-v0.1-6.7B.out &
