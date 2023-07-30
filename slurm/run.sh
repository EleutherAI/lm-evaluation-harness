#!/bin/bash

export PIP_INDEX_URL=http://pypi-cache/index
export PIP_TRUSTED_HOST=pypi-cache
export PIP_NO_CACHE=true
export HUGGINGFACE_CACHE_DIR=/ds/models/llms/cache
export HUGGINGFACE_HUB_CACHE=/ds/models/llms/cache
export HF_DATASETS_CACHE=/netscratch/juneja/cache/hf_datasets


cd /netscratch/juneja/projects/lm-evaluation-harness
pip install -e .


python main.py \
    --model llm-tools \
    --tasks atis \
