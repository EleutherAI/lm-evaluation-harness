#!/bin/bash

export PIP_INDEX_URL=http://pypi-cache/index
export PIP_TRUSTED_HOST=pypi-cache
export PIP_NO_CACHE=true
export HUGGINGFACE_CACHE_DIR=/ds/models/llms/cache
export HUGGINGFACE_HUB_CACHE=/ds/models/llms/cache
export HF_DATASETS_CACHE=/netscratch/juneja/cache/hf_datasets


cd /netscratch/juneja/projects/lm-evaluation-harness
pip install -e .


#python -m debugpy --listen serv-9225.kl.dfki.de:5678 main.py \
python main.py \
    --model llm-tools \
    --tasks atis \

# python main.py \
#     --model hf \
#     --model_args pretrained=EleutherAI/gpt-j-6B \
#     --tasks hellaswag \
#     --device cuda:0 \

# python -m scripts.write_out \
#     --output_base_path output \
#     --tasks toxigen \
#     --sets test \
#     --num_fewshot 1 \
#     --num_examples 2 \
