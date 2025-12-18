#!/bin/bash
#SBATCH --job-name=eval_nl2foam
#SBATCH --output=logs/eval_ablation_%j.log
#SBATCH --error=logs/eval_ablation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=100G
#SBATCH --time=50:00:00
#SBATCH --nodelist=p-1-cluster-node-002

source .venv/bin/activate


##########################################################################################################################
# Evaluate models trained on original data format and base model on eval set of original data format
##########################################################################################################################

echo "Evaluating model: /scratch/hieu/nl2foam_sft_0/outputs/nl2foam_sft_0__8__1765350318/"

lm_eval run --model vllm --model_args pretrained=/scratch/hieu/nl2foam_sft_0/outputs/nl2foam_sft_0__8__1765350318,tensor_parallel_size=8 --tasks nl2foam_gen_original nl2foam_llm_judge_original --gen_kwargs max_gen_toks=4096 --batch_size auto --output_path all-results-original/ --log_samples

lm_eval run --model vllm --model_args pretrained=/scratch/hieu/nl2foam_sft_0/outputs/nl2foam_sft_0__8__1765350318,tensor_parallel_size=8 --tasks nl2foam_perplexity_original --batch_size auto --output_path all-results-original/

echo "Done"
echo "--------------------------------"


echo "Evaluating model YYgroup/AutoCFD-7B"

lm_eval run --model vllm --model_args pretrained=YYgroup/AutoCFD-7B,tensor_parallel_size=4 --tasks nl2foam_gen_original nl2foam_llm_judge_original --gen_kwargs max_gen_toks=4096 --batch_size auto --output_path all-results-original/ --log_samples

lm_eval run --model vllm --model_args pretrained=YYgroup/AutoCFD-7B,tensor_parallel_size=4 --tasks nl2foam_perplexity_original --batch_size auto --output_path all-results-original/

echo "Done"
echo "--------------------------------"


echo "Evaluating model Qwen/Qwen3-8B"

lm_eval run --model vllm --model_args pretrained=Qwen/Qwen3-8B,tensor_parallel_size=8 --tasks nl2foam_gen_original nl2foam_llm_judge_original --gen_kwargs max_gen_toks=4096 --batch_size auto --output_path all-results-original/ --log_samples

lm_eval run --model vllm --model_args pretrained=Qwen/Qwen3-8B,tensor_parallel_size=8 --tasks nl2foam_perplexity_original --batch_size auto --output_path all-results-original/

echo "Done"
echo "--------------------------------"






##########################################################################################################################
# Evaluate models trained on formatted data and base model on eval set of formatted data (so no YYgroup/AutoCFD-7B as it's trained on original data format)
##########################################################################################################################



echo "Evaluating model: /scratch/hieu/nl2foam_sft_0/outputs/nl2foam_sft_0__8__1765777563/"

lm_eval run --model vllm --model_args pretrained=/scratch/hieu/nl2foam_sft_0/outputs/nl2foam_sft_0__8__1765777563,tensor_parallel_size=8 --tasks nl2foam_gen_formatted nl2foam_llm_judge_formatted --gen_kwargs max_gen_toks=4096 --batch_size auto --output_path all-results-formatted/ --log_samples

lm_eval run --model vllm --model_args pretrained=/scratch/hieu/nl2foam_sft_0/outputs/nl2foam_sft_0__8__1765777563,tensor_parallel_size=8 --tasks nl2foam_perplexity_formatted --batch_size auto --output_path all-results-formatted/

echo "Done"
echo "--------------------------------"


echo "Evaluating model Qwen/Qwen3-8B"

lm_eval run --model vllm --model_args pretrained=Qwen/Qwen3-8B,tensor_parallel_size=8 --tasks nl2foam_gen_formatted nl2foam_llm_judge_formatted --gen_kwargs max_gen_toks=4096 --batch_size auto --output_path all-results-formatted/ --log_samples

lm_eval run --model vllm --model_args pretrained=Qwen/Qwen3-8B,tensor_parallel_size=8 --tasks nl2foam_perplexity_formatted --batch_size auto --output_path all-results-formatted/

echo "Done"
echo "--------------------------------"

