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

echo "Evaluating model: /scratch/hieu/nl2foam_sft_0/outputs/nl2foam_sft_0__8__1765350318/"

lm_eval run --model vllm --model_args pretrained=/scratch/hieu/nl2foam_sft_0/outputs/nl2foam_sft_0__8__1765350318,tensor_parallel_size=8 --tasks nl2foam_gen nl2foam_llm_judged --gen_kwargs max_gen_toks=4096 --batch_size auto --output_path all-results/ --log_samples

lm_eval run --model vllm --model_args pretrained=/scratch/hieu/nl2foam_sft_0/outputs/nl2foam_sft_0__8__1765350318,tensor_parallel_size=8 --tasks nl2foam_perplexity --batch_size auto --output_path all-results/

echo "Done"
echo "--------------------------------"


echo "Evaluating model YYgroup/AutoCFD-7B"
lm_eval run --model vllm --model_args pretrained=YYgroup/AutoCFD-7B,tensor_parallel_size=4 --tasks nl2foam_gen nl2foam_llm_judged --gen_kwargs max_gen_toks=4096 --batch_size auto --output_path all-results/ --log_samples

lm_eval run --model vllm --model_args pretrained=YYgroup/AutoCFD-7B,tensor_parallel_size=4 --tasks nl2foam_perplexity --batch_size auto --output_path all-results/


echo "Done"
echo "--------------------------------"


echo "Evaluating model Qwen/Qwen3-8B"
lm_eval run --model vllm --model_args pretrained=Qwen/Qwen3-8B,tensor_parallel_size=8 --tasks nl2foam_gen nl2foam_llm_judged --gen_kwargs max_gen_toks=4096 --batch_size auto --output_path all-results/ --log_samples

lm_eval run --model vllm --model_args pretrained=Qwen/Qwen3-8B,tensor_parallel_size=8 --tasks nl2foam_perplexity --batch_size auto --output_path all-results/


echo "Done"
echo "--------------------------------"

