# Use the following instruction to specify task and mode
# $ bash benchmark.sh {Is PoE or not, T/F} {Path to the model} {Backbone name} {Name of benchmark} {Output path prefix} {is_api, T/F}

###################################################################

# Fetch the infomation from machine
PID=$(sacct -X --start now-3hours -o jobid | tail -n 1 | tr -d -c 0-9 )
GPU_NUM=$(($(rocm-smi --csv --showuniqueid | tail -n 2 | tr -d -c 0-9 | cut -c 1) + 1))

echo "PID=$PID, GPU_NUM=$GPU_NUM"

###################################################################

# Configure device
if [ "$GPU_NUM" == "4" ]
then 
    DEVICE="0,1,2,3"
elif [ "$GPU_NUM" ==  "8" ]
then
    DEVICE="0,1,2,3,4,5,6,7"
else # Default to single GPU evaluation
    DEVICE="0"
fi

###################################################################

# Configure model args
if [ "$1" == "T" ]
then
    MODEL_ARGS="is_poe=True,backbone=$3,pretrained=$2"
elif [ "$1" == "F" ]
then
    MODEL_ARGS="pretrained=$2"
fi

# Configure parallelize
if [ "$DEVICE" != "0" ] 
then
    MODEL_ARGS="parallelize=True,$MODEL_ARGS"
fi

# Set trust remote code
export HF_DATASETS_TRUST_REMOTE_CODE=1

###################################################################

# Configure tasks and its batchsize
TASK=$4
FEWSHOT_NUM=0
if [ "$TASK" == "openbookqa" ]
then
    BATCH_SIZE="auto"

elif [ "$TASK" = "winogrande" ]
then
    BATCH_SIZE="auto"

elif [ "$TASK" = "gpqa" ]
then
    TASK=gpqa_main_zeroshot
    BATCH_SIZE="auto"

elif [ "$TASK" = "hellaswag" ]
then
    BATCH_SIZE="auto"

elif [ "$TASK" = "mmlu" ]
then
    BATCH_SIZE="auto"

elif [ "$TASK" = "mmlu_5shots" ]
then
    TASK=mmlu
    FEWSHOT_NUM=5
    BATCH_SIZE="8"

elif [ "$TASK" = "gsm8k" ]
then
    TASK=gsm8k_cot
    FEWSHOT_NUM=8
    BATCH_SIZE="32"

elif [ "$TASK" = "triviaqa" ]
then
    BATCH_SIZE="32"

else
    echo "Task not found"
    exit 1
fi

###################################################################

# Configure output path
if [ "$FEWSHOT_NUM" != "0" ]
then
    OUTPUT_PATH="result/$5-$TASK-$FEWSHOT_NUM-shots.json"
else
    OUTPUT_PATH="result/$5-$TASK.json"
fi

###################################################################

# Append infomation to panel
echo "$PID,$5,$4,$BATCH_SIZE,$GPU_NUM" >> $WORK/panel

###################################################################
 
echo "========================================================================================"
echo "Start the evaluation with instruction below:"
echo "  HIP_VISIBLE_DEVICES=$DEVICE"
echo "    lm_eval --model hf"
echo "    --model_args $MODEL_ARGS"
echo "    --tasks $TASK"
echo "    --num_fewshot $FEWSHOT_NUM"
echo "    --batch_size $BATCH_SIZE"
echo "    --output_path $OUTPUT_PATH"
echo "========================================================================================"

HIP_VISIBLE_DEVICES=$DEVICE \
    lm_eval --model hf \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $FEWSHOT_NUM \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_PATH \
    --trust_remote_code \
    --overwrite_existing_result

exit 0
