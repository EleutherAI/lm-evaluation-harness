#!/bin/bash
#SBATCH --account="stablegpt"
#SBATCH --job-name="jp-eval-abeja2.7b-jsquad"
#SBATCH --partition=g40
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=11G
#SBATCH --output=/fsx/home-mkshing/slurm_outs/%x_%j.out
#SBATCH --error=/fsx/home-mkshing/slurm_outs/%x_%j.err

source /fsx/home-mkshing/venv/nlp/bin/activate
# MODEL_ARGS="pretrained=abeja/gpt-neox-japanese-2.7b,low_cpu_mem_usage=True"
MODEL_ARGS="pretrained=rinna/japanese-gpt-1b,use_fast=False"
# MODEL_ARGS="pretrained=naclbit/gpt-j-japanese-6.8b,low_cpu_mem_usage=True" <- hasn't released the weight yet
# MODEL_ARGS="pretrained=/fsx/jp-llm/hf_model/test,tokenizer=/fsx/home-mkshing/models/novelai-tokenizer,use_fast=False"
TASK="jsquad" # jsquad, jaquad, jcommonsenseqa, lambada_openai_mt_ja
 # num_fewshot:: jsquad/jaquad -> 2, jcommonsenseqa -> 3, lambada_openai_mt_ja -> 0
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot 2 \
    --device "cuda" \
    --no_cache

# [Example] From gpt-neox's checkpoints
# python ./deepy.py evaluate.py \
#     -d configs /fsx/jp-llm/ckpts/1b_tok=nai_data=mc4-cc100-wiki_bs=4m_tp=1_pp=1_init=wang-small-init_dtype=int64/global_step40000/configs/stable-lm-jp-1b-nai_tok-mc4_cc100_wiki.yml \
#     --eval_tasks lambada_openai_mt_ja \
#     --eval_num_fewshot 2

# # srun --account="stablegpt" --partition=g40 --gpus=1 --cpus-per-gpu=12 --mem-per-cpu=11G --job-name="jp_eval" --pty bash -i
# python main.py \
#     --model hf-causal \
#     --model_args "pretrained=/fsx/jp-llm/hf_model/test,tokenizer=/fsx/home-mkshing/models/novelai-tokenizer,use_fast=False" \
#     --tasks lambada_openai_mt_ja \
#     --device "cuda"