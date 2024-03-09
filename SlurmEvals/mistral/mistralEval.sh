#!/bin/bash
#SBATCH --ntasks-per-node=1               # Run one task
#SBATCH --partition=gen                   # Select a CPU partition
#SBATCH --mem=47GB                        # Expected amount of CPU RAM needed
#SBATCH --cpus-per-task=8                 # Use 8 CPU cores
#SBATCH --time=64:00:00 
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=tyler.liddell@city.ac.uk

# Enable modules command
source /opt/flight/etc/setup.sh
flight env activate gridware

# Remove any unwanted modules
module purge

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

python ~/lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args pretrained=/users/adbt150/archive/Mistral-7B-v0.1 \
            --batch_size 5 \
            --num_fewshot 0 \
            --tasks truthfulqa_gen \
            --write_out

            