export HF_DATASETS_CACHE="/net/nfs/mosaic/seanw/.cache"
export TRANSFORMERS_CACHE="/net/nfs/mosaic/seanw/.cache"

FEWSHOT=5
OUTPUT_DIR="output/lila"
mkdir -p ${OUTPUT_DIR}

TASKS="lila_addsub,lila_svamp_structured,lila_multiarith,lila_GSM8k_structured,lila_deepmind_mathematics_algebra,lila_MATH_algebra_crowdsourced"
python main.py --model_args pretrained=EleutherAI/pythia-12b-deduped --num_fewshot ${FEWSHOT} --model hf-causal --use_accelerate --tasks ${TASKS} --output_path ${OUTPUT_DIR}/pythia12bdeduped.json --batch_size 1 >& ${OUTPUT_DIR}/pythia12bdeduped.out &

