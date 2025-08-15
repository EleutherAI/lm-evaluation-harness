# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lm_eval.tasks.hallulens.utils import generate, jsonify_ans
import lm_eval.tasks.hallulens.prompt_template_nonsense as prompt_templates
import os
import json

class NonsenseNameEval:
    def __init__(self, evaluator_model, evaluator_tokenizer):
        self.evaluator_model = evaluator_model
        self.evaluator_tokenizer = evaluator_tokenizer

    def run_eval_generated(self, generation, name, _type, place=None):
        if place is not None:
                _place=" in " + place
        else:
            _place = ""
        abstain_prompt = prompt_templates.ABSTAIN_PROMPT_PLACE_NONSENSE.format(name=name,
            TYPE=_type,
            PLACE=_place,
            generation=generation,
        )

        abstains_eval_raw = generate(abstain_prompt, self.evaluator_model, self.evaluator_tokenizer)

        abstains_eval = jsonify_ans(
            raw_response=abstains_eval_raw,
            eval_prompt=abstain_prompt,
            key="does_believe",
            model=self.evaluator_model,
            tokenizer=self.evaluator_tokenizer
        )


        abstains_eval_res = []
        for o in abstains_eval:
            try:
                abstains_eval_res.append(not o['does_believe'])
            except:
                print(f"Error in eval_answer: {o}")
                exit()

        if len(abstains_eval_res) == 0:
            #print("No valid response received from the evaluator.")
            return {"abstention": None}
        return {"abstention": abstains_eval_res[0]}

class NonsenseMixedEval(NonsenseNameEval):
    def __init__(self, eval_model, eval_tokenizer):

        self.eval_model = eval_model
        self.eval_tokenizer = eval_tokenizer

    def run_eval_mixed(self, generation, prompt, _type, name):
        JSON_KEY = "does_believe"

        eval_prompts = {
            'medicine' : prompt_templates.ABSTAIN_PROMPT_NONSENSE_MEDICINE,
            'animal' : prompt_templates.ABSTAIN_PROMPT_ANIMAL_NONSENSE,
            'plant' : prompt_templates.ABSTAIN_PROMPT_ANIMAL_NONSENSE,
            'bacteria' : prompt_templates.ABSTAIN_PROMPT_ANIMAL_NONSENSE,
        }

        abstain_prompt = eval_prompts.get(_type).format(
                    prompt=prompt,
                    name=name,
                    generation=generation,
                )

        abstains_eval_raw = generate(abstain_prompt, self.eval_model, self.eval_tokenizer)

        abstains_eval = jsonify_ans(raw_response=abstains_eval_raw, eval_prompt=abstain_prompt, key=JSON_KEY,
                                    model=self.eval_model, tokenizer=self.eval_tokenizer)
        abstains_eval_res = []
        for o in abstains_eval:
            abstains_eval_res.append(not o[JSON_KEY])
        
        if len(abstains_eval_res) == 0:
            #print("No valid response received from the evaluator.")
            return {"abstention": None}
        return {"abstention": abstains_eval_res[0]}