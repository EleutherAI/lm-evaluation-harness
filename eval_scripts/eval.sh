# Load modules
module purge
module load hpcfund
module load rocm
module load pytorch

# [Important] Activate virtual environment and import packages
source ./eval/bin/activate
export PYTHONPATH="$WORK/eval/lib64/python3.9/site-packages:$PYTHONPATH"

# [Important] Import the Pool of Experts configuration here
export PYTHONPATH="$WORK/Pool-of-Experts/poe-patch:$PYTHONPATH"

# Configure transformer cache in $WORK area (so that it has larger space)
export TRANSFORMERS_CACHE="$WORK/.cache"

# Use API
# HIP_VISIBLE_DEVICES=$DEVICE \ 
#     lm_eval --model local-completions \
#     --model_args model=llama_test,base_url=http://localhost:8888/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,batch_size=16 \
#     --tasks winogrande \
#     --output_path $OUTPUT_PATH \
#     --trust_remote_code \
#     --overwrite_existing_result

# Example evaluation: without PoE
# BENCHMARK=hellaswag
# ARCH=olmoe
# BACKBONE=olmoe
# MODEL_NAME=allenai/OLMoE-1B-7B-0924
# bash lm-evaluation-harness-PoE/eval_scripts/benchmark.sh F $MODEL_NAME $BACKBONE $BENCHMARK $ARCH-base

# Example evaluation: with PoE

BENCHMARK=openbookqa
BACKBONE=tinyllama
MODEL_NAME=TinyLlama/TinyLlama_v1.1
bash lm-evaluation-harness-PoE/eval_scripts/benchmark.sh T $MODEL_NAME $BACKBONE $BENCHMARK tinyllama-untrained
