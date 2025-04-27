export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN
export HF_HUB_ENABLE_HF_TRANSFER=1

# rm -rf results logs
mkdir -p logs
mkdir -p results

## Load from Checkpoint
bash dp.sh /datadisk/storage/varunartifacts/containers/indic-phi/checkpoints/llama_bactrian/checkpoint-123 results > logs/llama_bactrian.log 2>&1
bash dp.sh /datadisk/storage/varunartifacts/containers/indic-phi/checkpoints/gemma_bactrian/checkpoint-81 results > logs/gemma_bactrian.log 2>&1
bash dp.sh /datadisk/storage/varunartifacts/containers/indic-phi/checkpoints/llama_orca/checkpoint-294 results > logs/llama_orca.log 2>&1
bash dp.sh /datadisk/storage/varunartifacts/containers/indic-phi/checkpoints/gemma_orca/checkpoint-84 results > logs/gemma_orca.log 2>&1

## HF Baselines
bash dp.sh "krutrim-ai-labs/Krutrim-2-instruct" results > logs/krutrim_2.log 2>&1

bash dp.sh "meta-llama/Llama-3.2-1B-Instruct" results > logs/llama_3.2_3b.log 2>&1
bash dp.sh "meta-llama/Llama-3.2-3B-Instruct" results > logs/llama_3.2_3b.log 2>&1
bash dp.sh "meta-llama/Llama-3.1-8B-Instruct" results > logs/llama_3.1_8b.log 2>&1
bash dp.sh "meta-llama/Llama-3.3-70B-Instruct" results > logs/llama_3.3_70b.log 2>&1
bash dp_mp.sh "meta-llama/Llama-3.1-405B-Instruct" results > logs/llama_3.1_405b.log 2>&1

bash dp.sh "google/gemma-3-1b-it" results > logs/gemma_3_1b.log 2>&1
bash dp.sh "google/gemma-3-4b-it" results > logs/gemma_3_4b.log 2>&1
bash dp.sh "google/gemma-3-12b-it" results > logs/gemma_3_12b.log 2>&1
bash dp.sh "google/gemma-3-27b-it" results > logs/gemma_3_27b.log 2>&1

bash dp.sh "microsoft/phi-4-mini-instruct" results > logs/phi_4_mini.log 2>&1
bash dp.sh "microsoft/phi-4" results > logs/phi_4.log 2>&1

bash dp.sh "CohereLabs/aya-expanse-8b" results > logs/aya_expanse_8b.log 2>&1
bash dp.sh "CohereLabs/aya-expanse-32b" results > logs/aya_expanse_32b.log 2>&1
