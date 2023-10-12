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
flight env activate gridware

#Remove any unwanted modules
module purge
module load libs/nvidia-cuda/11.2.0/bin
module load /users/adbt150/yes

source ~/yes/etc/profile.d/conda.sh
conda activate llm
nvidia-smi
GPUS_PER_NODE=1
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
export MASTER_PORT=8214
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0

python ~/lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args pretrained=/users/adbt150/archive/Llama-2-7b-hf \
            --batch_size 16 \
            --device cuda:0 \
            --num_fewshot 5 \
            --tasks hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science \
            --shuffle unigram