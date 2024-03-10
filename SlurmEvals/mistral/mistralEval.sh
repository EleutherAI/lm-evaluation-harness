#!/bin/bash
#SBATCH --ntasks-per-node=1                        # Run one task
#SBATCH --partition=gengpu                       # Select the correct partition.                              # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                      # Use 8 cores, most of the procesing happens on the GPU
#SBATCH --mem=47GB                                  # Expected ammount CPU RAM needed (Not GPU Memory)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6                          # Use 4 cores, most of the procesing happens on the GPU
#SBATCH --time=64:00:00 
#SBATCH --mail-type=BEGIN,END,FAIL                       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=tyler.liddell@city.ac.uk   # Where to send mail

#Enable modules command
source /opt/flight/etc/setup.sh
#flight env activate gridware

#Remove any unwanted modules
module purge
#module load libs/nvidia-cuda/11.2.0/bin

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm
#nvidia-smi
#GPUS_PER_NODE=1
# Number of GPU workers, for single-worker training, please set to 1
#WORKER_CNT=1
export MASTER_PORT=8214
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0

python ~/lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args pretrained=/users/adbt150/archive/Mistral-7B-v0.1 \
            --batch_size 5 \
            --device cuda:0 \
            --num_fewshot 0 \
            --tasks truthfulqa_gen \
            --write_out
