# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from lm_eval.tasks.hallulens.utils import generate
from typing import List
from tqdm.contrib.concurrent import thread_map


def print_all_metrics(final_results_df, k=32):
    overall_recall = final_results_df.groupby("prompt").recall.first().mean()
    overall_precision = final_results_df.groupby("prompt").precision.first().mean()
    overall_f1 = final_results_df.groupby("prompt").f1.first().mean()
    final_results_df["overall_recall"] = overall_recall
    final_results_df["overall_precision"] = overall_precision
    final_results_df["overall_f1"] = overall_f1

    med_n_claims = final_results_df.groupby("prompt").n_claims.first().median()
    print("Precision:", "%.3f" % overall_precision)
    print(f"Recall@{k}:", "%.3f" % overall_recall)
    print(f"F1@{k}", "%.3f" % overall_f1)
    print(f"med_n_claims", "%.3f" % med_n_claims)

def calculate_all_metrics(final_results_df, k=32):
    final_results_df["precision"] = final_results_df.groupby("prompt").is_supported.transform(
        "mean"
    )
    final_results_df["recall"] = final_results_df.groupby("prompt").is_supported.transform(
        lambda g: min(g.sum() / k, 1)
    )
    final_results_df = (
        final_results_df.groupby("prompt", group_keys=False)
        .apply(f1_score) #, include_groups=False) # NOTE: python 3.9 < 
        .reset_index()
    )
    final_results_df["k"] = k
    final_results_df["n_claims"] = final_results_df.groupby("prompt").is_supported.transform(
        "count"
    )

    overall_recall = final_results_df.groupby("prompt").recall.first().mean()
    overall_precision = final_results_df.groupby("prompt").precision.first().mean()
    overall_f1 = final_results_df.groupby("prompt").f1.first().mean()

    final_results_df["overall_recall"] = overall_recall
    final_results_df["overall_precision"] = overall_precision
    final_results_df["overall_f1"] = overall_f1

    med_n_claims = final_results_df.groupby("prompt").n_claims.first().median()

    print("Precision:", "%.3f" % overall_precision)
    print(f"Recall@{k}:", "%.3f" % overall_recall)
    print(f"F1@{k}", "%.3f" % overall_f1)
    print(f"med_n_claims", "%.3f" % med_n_claims)
    
    return final_results_df

def f1_score(g):
    prec = g.precision.iloc[0]
    rec = g.recall.iloc[0]
    if (prec + rec) == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    g["f1"] = f1
    return g

def remove_prefix(text:str , prefix:str):
    # for < python 3.9 
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def parse_claim_extraction(claim_extraction: List[str]):
    res = []
    for claim in claim_extraction:
            if not claim.startswith('- ') or \
                remove_prefix(claim, '- ').strip() == '':
                continue
            claim_text = remove_prefix(claim, '- ').strip()
    
            if claim_text== "No available facts" or \
                claim_text == "No available facts.":
                continue

            res.append(claim_text)
    return res

def save_eval_raw(
        raw_eval_list: List[str],
        output_file):
    
    with open(output_file, "w") as f:
        for r in raw_eval_list:
            f.write(json.dumps({"eval_res": r}) + "\n")

def read_eval_raw(eval_raw_file):
    eval_raw_res = []
    if os.path.exists(eval_raw_file):
        with open(eval_raw_file, "r") as f:
            eval_raw_res = [json.loads(line)["eval_res"] for line in f]
    return eval_raw_res
    
# def model_eval_step(evaluator, prompts, max_token=512, batch_size=16, max_workers=16, api_i=0):
    
    
#     eval_raw_res = thread_map(
#             lambda p: lm.generate(p, evaluator, i=api_i),
#             prompts,
#             max_workers=max_workers,
#             desc=f"using vllm {evaluator}")
#     return eval_raw_res

def jsonify_ans(raw_responses, eval_prompts, model, tokenizer, key):

    def check_validity(gen):
        if '{{"{}":false}}'.format(key) in gen.lower():
            return '{{"{}":false}}'.format(key)
        elif '{{"{}":true}}'.format(key) in gen.lower():
            return '{{"{}":true}}'.format(key)
        else:
            return -1
        
    jsonifyed_res  = []
    for r, p in zip(raw_responses, eval_prompts):
        
        if check_validity(r) != -1:
            jsonifyed_res.append(json.loads(check_validity(r)))
            continue
        else:
            r = r.split("\n")[0]
            try:
                jsonifyed_res.append(json.loads(r))
            except:
                print(f"Error in eval_answer: {r}")
                error = True
                error_count = 0
                
                while error:
                    re_eval = generate(p, model, tokenizer)

                    try: 
                        print("\n** RETRY:", re_eval)
                        if check_validity(re_eval) != -1:
                            json_res = json.loads(check_validity(re_eval))
                        else:
                            re_eval = re_eval.split("\n")[0]
                            json_res = json.loads(re_eval)
                        error = False
                        
                    except:
                        print("*** trying again** \n")
                        error = True
                    error_count += 1

                    if error_count > 3:
                        print("Error count exceeded 3. Skipping this prompt.")
                        jsonifyed_res.append({"error": "Error count exceeded 3. Skipping this prompt."})
                        break
                jsonifyed_res.append(json_res)
                print("<<< PASS >>>")

    return jsonifyed_res
