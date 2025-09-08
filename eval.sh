#! /bin/bash

SEED=42
# BATCH_SIZE=auto
BATCH_SIZE=8 #you may consider changing this based on your GPU memory
DEVICE="cuda:0"
OVERWRITE=false

RESULTS_PATH="/home/mattia_llm/Desktop/culture/lm-evaluation-harness/results" #change this to your path

BASE_CHECKPOINTS=(
)
IFT_CHECKPOINTS=(
    "ALLaM-AI/ALLaM-7B-Instruct-preview"
)
# BASE_CHECKPOINTS + IFT_CHECKPOINTS
CHECKPOINTS=("${IFT_CHECKPOINTS[@]}" "${BASE_CHECKPOINTS[@]}")

BENCHMARKS=(
    QASI_subtask1
)

# import HF token
# export HF_TOKEN=$(grep 'hf_token:' /mnt/data/users/anwarvic/mbzuai_exps/creds.yaml | awk '{print $2}' | tr -d '"')
# export HF_HOME="/mnt/data/users/anwarvic/.cache/huggingface" # if you are using the default cache path, you can remove this line

for BENCHMARK in "${BENCHMARKS[@]}"; do
    for CHECKPOINT in "${CHECKPOINTS[@]}"; do
        MODEL_NAME=$(basename "${CHECKPOINT}")
        # highlight the model name and benchmark for clarity
        echo -e "Evaluating \e[1;34m${MODEL_NAME}\e[0m on \e[1;34m${BENCHMARK}\e[0m Benchmark\n"
        # echo -e "\n\e[1;34mEvaluating model: ${MODEL_NAME}\e[0m on benchmark: ${BENCHMARK}\n"
        OUT_PATH="${RESULTS_PATH}/${BENCHMARK}/${MODEL_NAME}"
        mkdir -p "${OUT_PATH}"

        # Skip evaluation if results file exists and override is false
        if [ "$OVERWRITE" = false ] && [ "$(ls -A "${OUT_PATH}")" ]; then
            echo -e "Skipping (results already exist)"
            continue
        fi

        # empty cuda cache
        echo -e "Emptying CUDA cache..."
        python -c "import torch; torch.cuda.empty_cache()"
      
        # if the checkpoint doesn't start with mistral
        if [[ ${CHECKPOINT} == *"mistral"* ]]; then
            MAX_TOKENS=2048
            PREFIX="Answer the following question directly. No explanations and no introductions. Only provide the answer."
            CONCURRENCY=1
            lm-eval \
                --model "mistral-completions" \
                --model_args "model_name=${CHECKPOINT},max_tokens=${MAX_TOKENS},prompt=${PREFIX},num_concurrent=${CONCURRENCY}" \
                --tasks ${BENCHMARK} \
                --log_samples \
                --output_path "${OUT_PATH}/results.json" \
                --seed ${SEED} \
                --verbosity DEBUG
        else
            # add --apply_chat_template in for IFT_CHECKPOINTS
            APPLY_CHAT_TEMPLATE=""
            if [[ " ${IFT_CHECKPOINTS[@]} " =~ " ${CHECKPOINT} " ]]; then
                echo -e "Applying chat template for IFT checkpoint: ${CHECKPOINT}"
                APPLY_CHAT_TEMPLATE="--apply_chat_template"
            fi
            # Use the `lm-eval` command for evaluation
            lm-eval \
                --model hf \
                --model_args "pretrained=${CHECKPOINT}" \
                --tasks ${BENCHMARK} \
                --log_samples \
                --batch_size ${BATCH_SIZE} \
                ${APPLY_CHAT_TEMPLATE} \
                --output_path "${OUT_PATH}/results.json" \
                --seed ${SEED} \
                --device ${DEVICE} \
                --verbosity DEBUG
        fi
    done
done
