# Usage:
# sh mc-answer-prompt-experiment.sh \
#   -e <engine> \
#   -k <number of examples> \
#   -s <mc-setting = "freeform" | "option" | "letter" | "number"> \

while getopts e:k:s: flag
do
    case "${flag}" in
        e) engine=${OPTARG};;
        k) k_shot=${OPTARG};;
        s) setting=${OPTARG};;
    esac
done

ENGINE=$engine
KSHOT=$k_shot
MC_SETTING=$setting

# Set environment variables.
#export GOOSEAI_API_SECRET_KEY=sk-
export MC_SETTING=$setting

# Setup paths.
RESULT_DIR=$(pwd)/mc-task-results/$ENGINE/$KSHOT-shot
mkdir -p $RESULT_DIR
export QUESTION_RESULT_PATH=$RESULT_DIR/$MC_SETTING
mkdir -p $RESULT_DIR/$MC_SETTING

# Tasks to run.
HENDRYCKS_TEST=hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions

# Runner function.
run_experiment(){
    local curr_engine=$1
    local setting=$2
    local output_path=$RESULT_DIR/$setting
    
    # Log stuff.
    echo "\n"
    echo "###################################################"
    echo "PID: $PPID"
    echo "MC Setting: $setting"
    echo "Few-shot: $KSHOT"
    echo "Current Engine: $curr_engine"
    echo "Current Results Dir:\n$output_path"
    echo "Start Time: $(date)"
    echo "###################################################"
    echo "\n"

    python3 -m scripts.write_out --output_base_path $output_path --tasks hendrycksTest-abstract_algebra --sets test --num_fewshot $KSHOT
    mv $output_path/hendrycksTest-abstract_algebra $output_path/example_prompt 

    python3 main.py \
        --model gooseai \
        --model_args engine=$curr_engine \
        --tasks $HENDRYCKS_TEST \
        --output_path $output_path/results.json \
        --num_fewshot $KSHOT

# Test Call.
#     python3 main.py \
#         --device cpu \
#         --model gpt2 \
#         --tasks anagrams1 \
#         --limit 2 \
#         --output_path $output_path/results.json
}

# Run experiment.
touch $RESULT_DIR/$MC_SETTING/out.log
run_experiment $ENGINE $MC_SETTING > $RESULT_DIR/$MC_SETTING/out.log

# Setup subshells?
# ()
