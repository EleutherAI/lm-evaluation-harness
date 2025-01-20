```
Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
````
# Non Greedy Evaluation

This task checks for model's consistency towards seed changes during generation.
More particularly it evaluates the model's accuracy and consistancy rate with 5
different seeds (seed = 1, 2,...,5) for a fixed prompt with temperature set to 0.7.

## How to run the Non-Greedy evaluation of SCORE?

Evaluation for non greedy tasks differs a bit from other score tasks as it is required to pass different seeds as an argument manually. Below you can find the step-by-step guide on how to correctly run the **Score Non-Greedy** evaluation.

To run the evaluation of the Non-Greedy tasks with 5 different seeds you should:
1. For a given dataset run the evaluation by
   * specifying the task as `score_non_greedy_robustness_{DATASET_NAME}` (`DATASET_NAME` being either`agieval`, `mmlu_pro` or `math`)
   * fixing the seed with the run argument `--seed=1`
   * passing the `--log_samples` argument*
   * specifying an output with `--output_path=SOME_OUTPUT_PATH/seed_1`
   * if running with vllm it is important to set the seed in the `--model_args` just by specifying the `seed` parameter\

2. Repeat the process for 5 times**, changing the `--seed` and the `--output_path` arguments accordingly from 1 to 5.

3. When all 5 runs are finished and logs are saved, run the `./lm_eval/tasks/score/non_greedy_summarizer.py` script by passing the the output directory of the above runs to the `--log_dir` argument***, and by specifying the dataset name for which the evaluations were run with `--dataset` argument(`agieval`, `mmlu_pro` or `math`). \

4. The script will return the default lm_evaluation_harness table where accuracies for each seed and the consistancy rate are calculated.


\* _As this evaluation requires `--log_samples` to be True, it will need some extra disk space to save the prediction results for each seed._

\*\* _Refer to [`./lm_eval/tasks/score/non_greedy.sh`](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/score/non_greedy.sh) to see an example of non greedy evaluation command for each seed._

\*\*\* _To `--log_dir` argument one should pass the path of the parent folder of `"seed_1", "seed_2", ...` directories, that is not necessarily the `--output_path` passed to the evaulater in the 1st step._
