import json
import re
from typing import List


def generate(prompt, model, tokenizer, temperature=0.0, top_p=1.0, max_tokens=512):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        #max_len = min(tokenizer.model_max_length, 4096)  # defensive coding
        #input = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        input = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        if temperature > 0.0:
            output = model.generate(
                input_ids=input["input_ids"],
                attention_mask=input["attention_mask"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            # If temperature is 0, we use greedy decoding
            output = model.generate(
                input_ids=input["input_ids"],
                attention_mask=input["attention_mask"],
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        prompt_length = input["input_ids"].shape[1]
        result = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
        del input  # Free memory
        return result

def generate_batch(prompts, model, tokenizer, temperature=0.0, top_p=1.0, max_tokens=512):
    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    #max_len = min(tokenizer.model_max_length, 4096)
    #inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0.0 else None,
        top_p=top_p,
        do_sample=temperature > 0.0
    )

    prompt_lengths = (inputs["attention_mask"].sum(dim=1)).tolist()
    result = [
        tokenizer.decode(out[p_len:], skip_special_tokens=True)
        for out, p_len in zip(output, prompt_lengths)
    ]
    del inputs  # Free memory
    return result


def jsonify_ans_longwiki(raw_responses, eval_prompts, model, tokenizer, key):

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
                    re_eval = generate(
                        prompt=p,
                        model=model,
                        tokenizer=tokenizer,
                        temperature=0.0,
                        max_tokens=128
                    )

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

def parse_claim_extraction(claim_extraction: List[str]):
    res = []

    for claim in claim_extraction.split('\n'):
        if not claim.startswith('- ') or \
            remove_prefix(claim, '- ').strip() == '':
            continue
        claim_text = remove_prefix(claim, '- ').strip()

        if claim_text== "No available facts" or \
            claim_text == "No available facts.":
            continue

        res.append(claim_text)
    return res

def remove_prefix(text:str , prefix:str):
    # for < python 3.9 
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

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

def jsonify_ans(
        raw_response: List[str],
        eval_prompt: List[str], 
        key: str,
        model,
        tokenizer):

    def check_validity(gen):
        gen_nospace = gen.replace(" ", "")
        if '{{"{}":false}}'.format(key) in gen_nospace:
            return '{{"{}":false}}'.format(key)
        elif '{{"{}":true}}'.format(key) in gen_nospace:
            return '{{"{}":true}}'.format(key)
        else:
            return -1
        
    jsonifyed_res  = []

    if check_validity(raw_response) != -1:
        jsonifyed_res.append(json.loads(check_validity(raw_response)))

    else:
        raw_response = raw_response.split("\n")[0]
        try:
            jsonifyed_res.append(json.loads(raw_response))
        except:
            #print(f"Error in eval_answer: {raw_response}")
            error = True
            error_count = 0
            
            while error:
                re_eval = generate(eval_prompt, model, tokenizer)
                    
                try: 
                    #print("\n** RETRY, prompt is:", eval_prompt)
                    if check_validity(re_eval) != -1:
                        json_res = json.loads(check_validity(re_eval))
                    else:
                        json_res = json.loads(re_eval.split("\n")[0])
                    error = False
                    
                except:
                    #print("*** trying again** \n")
                    error = True
                error_count += 1

                if error_count > 3:
                    #print("Error count exceeded 3. Skipping this prompt.")
                    jsonifyed_res.append({"error": "Error count exceeded 3. Skipping this prompt."})
                    return []
            jsonifyed_res.append(json_res)
            print("<<< PASS >>>")
    return jsonifyed_res