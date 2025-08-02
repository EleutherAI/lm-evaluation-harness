# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import json

from dataclasses import dataclass
from typing import List, Optional, Dict

import pandas as pd
from tqdm import tqdm   

import lm_eval.tasks.hallulens.prompt_templates
from segtok.segmenter import split_single

from transformers import AutoTokenizer
import lm_eval.tasks.hallulens.utils
from lm_eval.tasks.hallulens.longwiki_retrieval import LongWikiRetrieval, LongWikiDB
import lm_eval.tasks.hallulens.longwiki_utils as utils

@dataclass
class Claim:
    claim: str
    sentence: object
    refernce: Optional[str] = None
    topic: Optional[str] =  None
    search_results: Optional[List[Dict[str, str]]] = None
    prompt: Optional[str] = None
    is_supported: Optional[bool] = None
    question: Optional[str] = None # same as generation.prompt

@dataclass
class Sentence:
    sentence: str
    generation: object
    prompt: Optional[str]
    claims: Optional[List[Claim]] = None


@dataclass
class Generation:
    generation: str
    prompt: str
    sentences: Optional[List[Sentence]] = None
    abstain: Optional[bool] = None
    reference: Optional[str] = None
    topic: Optional[str] = None
    def __hash__(self) -> int:
        return hash(self.generation + self.prompt)
    def __eq__(self, other) -> bool:
        return self.generation == other.generation and self.prompt == other.prompt

class FactHalu:
    def __init__(
            self,
            abstention_model,
            abstention_tokenizer,
            claim_extractor,
            claim_extractor_tokenizer,
            claim_verifier,
            claim_verifier_tokenizer,

            k: int = 32,
            db_path="/data/wiki_data/.cache/enwiki-20230401.db",
            args=None
        ):
        
        self.k = k
        self.db_path = db_path
        self.db = LongWikiDB(db_path=db_path)

        self.abstention_model = abstention_model
        self.abstention_tokenizer = abstention_tokenizer

        self.claim_extractor = claim_extractor
        self.claim_extractor_tokenizer = claim_extractor_tokenizer

        self.verifier = claim_verifier
        self.verifier_tokenizer = claim_verifier_tokenizer
    

    def run(self, prompt, generation, title, reference=None):
        """
        Evaluate longwiki from model error.
        Saves results to output_csv as jsonl with one line per prompt.
        """
        final_result = {
            "abstained": 0,
            "precision": None,
            "recall": None,
            "f1": None,
        }
        #initiate generation object
        _generation = Generation(
            generation=generation,
            prompt=prompt
        )
        if reference is not None:
            _generation.reference = reference
        _generation.topic = title

        ### [[STEP #1]] False Refusal Test
        _generation = self.eval_abstention(
            prompt=prompt,
            generation=generation,
        )

        if _generation.abstain:
            final_result["abstained"] = 1
            return final_result

        ### [[STEP #2]] Extract claims
        print("\n[[Step 2]] Extracting Claims starts")
        all_claims = self.extract_claims(
            generation=_generation,
            prompt=prompt
        )

        if _generation.abstain:
            final_result["abstained"] = 1
            return final_result

        ### [[STEP #3]] Verify claims
        print(f"\n[[Step 3]] Verifying Claims starts. Len: {len(all_claims)}")
        all_verification_responses = self.verify_claims(all_claims)

        for claim, verification_response in zip(all_claims, all_verification_responses):
            claim.is_supported = verification_response["is_supported"]                

        ### [[[ STEP #4]]] Calculate metrics: precision, recall@k, f1, response ratio
        
        print(f"[[Step 4]] Calculating metrics")
        final_results = []
        for sentence in generation.sentences:
            if not sentence.claims:
                final_results.append(
                    {
                        "prompt": generation.prompt,
                        "is_supported": None,
                        "claim": "no claims",
                        "sentence": sentence.sentence,
                        "title": generation.topic
                    }
                )
            else:
                for claim in sentence.claims:
                    final_results.append(
                        {
                            "prompt": generation.prompt,
                            "is_supported": claim.is_supported,
                            "claim": claim.claim,
                            "sentence": sentence.sentence,
                            "title": generation.topic

                        }
                    )
        final_results_df = pd.DataFrame(final_results)
        final_results_df = utils.calculate_all_metrics(final_results_df, k=self.k)
        overall_recall = final_results_df.groupby("prompt").recall.first().mean()
        overall_precision = final_results_df.groupby("prompt").precision.first().mean()
        overall_f1 = final_results_df.groupby("prompt").f1.first().mean()
        final_result["precision"] = overall_precision
        final_result["recall"] = overall_recall
        final_result["f1"] = overall_f1
        return final_result

        

##########################################################################################
##########################################################################################


    def eval_abstention(self, prompt, generation):
        abstain_prompt = prompt_templates.ABSTAIN_PROMPT.format(
            prompt=prompt.strip(), generation=generation
        ).strip()

        abstains_eval_raw = utils.generate(
            prompt=abstain_prompt,
            model=self.abstention_model,
            tokenizer=self.abstention_tokenizer,
            temperature=0.0,
            max_tokens=128,
        )

        abstains_eval = utils.jsonify_ans_longwiki(
            raw_responses=[abstains_eval_raw],
            eval_prompts=[abstain_prompt],
            model=self.abstention_model,
            tokenizer=self.abstention_tokenizer,
            key="is_knowledgeable"
        )

        evaluation = abstains_eval[0]
        return not evaluation["is_knowledgeable"]

    def extract_claims(self, generation, prompt):
        all_claim_extractions = []

        all_sentences = make_claim_extraction_prompts(
            generation=generation,
            prompt=prompt,
            tokenizer=self.claim_extractor_tokenizer
        )

        to_extract_prompts = [a.prompt for a in all_sentences]

        for prompt in to_extract_prompts:
            batch_results = utils.generate(prompt, self.claim_extractor, tokenizer=self.claim_extractor_tokenizer, max_tokens=512)
            all_claim_extractions.extend(batch_results)
        
        print("***** [2-2] Parsing extracted claims")
        all_claims = []
        deduplicate = set()
        assert len(all_claim_extractions) == len(all_sentences)

        for claim_extraction, sentence in zip(all_claim_extractions, all_sentences):
            if (not claim_extraction) or \
                claim_extraction.strip() == "No verifiable claim." or\
                claim_extraction.strip() == "No available facts" or \
                claim_extraction.strip() == "No available facts.":
                sentence.claims = []
                continue

            parsed_claim_extraction = utils.parse_claim_extraction(claim_extraction)
            
            sentence_claims = []
            for claim_text in parsed_claim_extraction:
                if (
                    claim_text.strip() != ""
                    and claim_text not in deduplicate
                ):
                    deduplicate.add(claim_text)
                    claim = Claim(claim=claim_text, \
                                  sentence=sentence, \
                                    refernce=sentence.generation.reference,\
                                    topic=sentence.generation.topic,\
                                    question=sentence.generation.prompt
                                ) 
                    sentence_claims.append(claim)
                    all_claims.append(claim)

            sentence.claims = sentence_claims

        # if deduplicate is empty, return empty list
        if not deduplicate:
            generation.abstain = True

        return all_claims

    def verify_claims(self, all_claims: List[Claim]):


        print("***** [3] Ref Src: ", self.ref_src)
        # 1. Prepare the prompt for verification
        retrieval = LongWikiRetrieval(self.db, cache_base_path=self.CACHE_BASE_PATH, embed_cache_path=self.embedded_cache_path, \
                            retrieval_type="gtr-t5-large", batch_size=64)
        questions = list(set([claim.question for claim in all_claims]))
        retrieval.make_ner_cache(questions)
        for claim in tqdm(all_claims):
            passages = retrieval.get_topk_related_passages(topic=claim.topic, claim=claim.claim, question=claim.question, k=5)
            context = ""
            for _, psg in enumerate(reversed(passages)):
                context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
            claim.prompt = prompt_templates.VERIFICATION_TEMPLATE_W_REFERENCE_RETRIEVAL.format(claim=claim.claim, reference=context)
        print("***** Prepared all verification prompts")

        # 2. Verify the claims
        verification_prompts = [c.prompt for c in all_claims]

        claim_verification_res = []
        for i in range(0, len(verification_prompts), 100):
            batch_prompts = verification_prompts[i:i+100]
            batch_results = utils.generate_batch(batch_prompts, self.verifier, tokenizer=self.verifier_tokenizer, max_tokens=512)
            claim_verification_res.extend(batch_results)


        assert len(claim_verification_res) == len(all_claims)
        # 3. post process the verification result
        claim_verification_results = utils.jsonify_ans_longwiki(raw_responses=claim_verification_res, \
                                                            eval_prompts=verification_prompts, 
                                                            evaluator=self.verification_model,
                                                            tokenizer=self.verifier_tokenizer,
                                                            key="is_supported")

        return claim_verification_results

def make_claim_extraction_prompts(generation, prompt, tokenizer):
    """
    Given a model output
    - split into sentences
    - go para by para, always add the first sent of the para into context1
    - snippet = (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
    Return list of {"prompt": prompt_text, "sentence": target_sentence}
    """
    sentences = []
    # split the text into sentences
    sentences_text = [x.strip() for x in split_single(generation)]
    question = prompt.replace("Answer in one paragraph.", "").strip()
    response = generation.strip()

    for i, sentence in list(enumerate(sentences_text)):
        if len(sentence) < 5:
            continue
        context1 = " ".join(sentences_text[max(0, i - 3) : i])
        target_sentence = sentences_text[i]
        sentence = f"<SOS>{target_sentence.strip()}<EOS>"
        context2 = " ".join(sentences_text[i + 1 : i + 2])
        snippet = f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()
        prompt_text = prompt_templates.EXTRACT_CLAIMS_TEMPLATE.format(
            snippet=snippet, sentence=sentence
        )
        # check token 
        prompt_len = len(tokenizer.encode(prompt_text))
        if prompt_len > 3500:
            context1 = " ".join(sentences_text[max(0, i - 2) : i])
            snippet = f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()
            
            prompt_text = prompt_templates.EXTRACT_CLAIMS_SHORT_TEMPLATE.format(
            snippet=snippet, sentence=sentence
            )   
            
            if len(tokenizer.encode(prompt_text)) > 3500:
                
                prompt_text = prompt_templates.EXTRACT_CLAIMS_EXTREME_SHORT_TEMPLATE.format(
                    snippet=snippet, sentence=sentence
                ) 
                
                if len(tokenizer.encode(prompt_text)) > 3500:
                    prompt_text = prompt_templates.EXTRACT_CLAIMS_EXTREME_EXTREME_SHORT_TEMPLATE.format(
                        snippet=snippet, sentence=sentence
                    )
                
            assert len(tokenizer.encode(prompt_text)) <= 3500

        sentences.append(
            Sentence(
                prompt=prompt_text, sentence=target_sentence, generation=generation
            )
        )

    return sentences