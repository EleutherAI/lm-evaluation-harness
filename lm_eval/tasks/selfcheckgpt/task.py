import os
from typing import Union, List


from lm_eval.api.task import Task
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task
from lm_eval.api.metrics import mean

import spacy
from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckNLI, SelfCheckBERTScore, SelfCheckNgram


@register_task("selfcheckgpt")
class SelfCheckGpt(Task):
    VERSION = 0.0
    DATASET_PATH = "potsawee/wiki_bio_gpt3_hallucination"
    DATASET_NAME = None
    OUTPUT_TYPE = 'generate_until'
    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, config=None):
        super().__init__(data_dir=data_dir, cache_dir=cache_dir, download_mode=download_mode, config=config)
        self.generation_kwargs = {"temperature": 0.0, "do_sample": False}
        self.generation_kwargs_sampling_number = 5 # the number of sampling for self-consistence
        self.generation_kwargs_sampling = {"temperature": 1.0, "do_sample": False}

        self.selfcheckgpt_type = os.environ.get('SELFCHECKGPTTYPE', 'SelfCheckNgram')
        self.selfcheckgpt_device = os.environ.get('SELFCHECKGPTDEVICE', 'cpu')
        self.selfcheckgpt_nlp = spacy.load("en_core_web_sm")

        if self.selfcheckgpt_type == 'SelfCheckNgram':
            self.selfcheckgpt = SelfCheckNgram(n=1)
        elif self.selfcheckgpt_type == 'SelfCheckBERTScore':
            self.selfcheckgpt = SelfCheckBERTScore(rescale_with_baseline=True)
        elif self.selfcheckgpt_type == 'SelfCheckMQAG':
            self.selfcheckgpt = SelfCheckMQAG(device=self.selfcheckgpt_device)
        elif self.selfcheckgpt_type == 'SelfCheckNLI':
            self.selfcheckgpt = SelfCheckNLI(device=self.selfcheckgpt_device)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["evaluation"]

    def doc_to_text(self, doc):
        doc_text = doc["wiki_bio_text"]
        doc_text = doc_text.split()
        doc_text = " ".join(doc_text[:5])
        doc_text = f"Please generating a Wikipedia passage starting with: {doc_text}\n"
        return doc_text

    def doc_to_target(self, doc):
        answer = doc['wiki_bio_text']
        return answer

    def construct_requests(
        self, doc: dict, ctx: str, **kwargs
    ) -> Union[List[Instance], Instance]:
        arguments = (ctx, self.generation_kwargs)
        request_list = [
            Instance(
                request_type=self.OUTPUT_TYPE,
                doc=doc,
                arguments=arguments,
                idx=0,
                **kwargs
                ),
        ]
        sampling_arguments = (ctx, self.generation_kwargs_sampling)
        request_list.extend([
            Instance(
                request_type=self.OUTPUT_TYPE,
                doc=doc,
                arguments=sampling_arguments,
                idx=idx,
                **kwargs
                )
            for idx in range(1, self.generation_kwargs_sampling_number+1)
            ]
        )
        return request_list


    def process_results(self, doc, results):
        response_temperature_0 = results[0]
        other_responses = results[1:]
        passage = self.doc_to_target(doc)

        sentences = self.selfcheckgpt_nlp(response_temperature_0)
        sentences = [sent.text.strip() for sent in sentences.sents]
        if self.selfcheckgpt_type == 'SelfCheckNgram':
            selfcheckgpt_scores = self.selfcheckgpt.predict(
                sentences = sentences,
                passage = response_temperature_0,
                sampled_passages = other_responses,
                )
            return {'avg-selfcheckgpt': selfcheckgpt_scores['doc_level']['avg_neg_logprob'],
                    'max-selfcheckgpt': selfcheckgpt_scores['doc_level']['avg_max_neg_logprob']}

        elif self.selfcheckgpt_type == 'SelfCheckBERTScore':
            selfcheckgpt_scores = self.selfcheckgpt.predict(
                sentences = sentences,
                sampled_passages = other_responses,
                )
        elif self.selfcheckgpt_type == 'SelfCheckMQAG':
            selfcheckgpt_scores = self.selfcheckgpt.predict(
                sentences = sentences,
                sampled_passages = other_responses,
                )
        elif self.selfcheckgpt_type == 'SelfCheckNLI':
            selfcheckgpt_scores = self.selfcheckgpt.predict(
                sentences = sentences,
                passage = response_temperature_0,
                sampled_passages = other_responses,
                num_questions_per_sent = 5,          # number of questions to be drawn
                scoring_method = 'bayes_with_alpha', # options = 'counting', 'bayes', 'bayes_with_alpha'
                beta1 = 0.8, beta2 = 0.8,            # additional params depending on scoring_method
                )

        selfcheckgpt_scores_avg = sum(selfcheckgpt_scores) / len(selfcheckgpt_scores) if len(selfcheckgpt_scores) > 0 else 0
        selfcheckgpt_scores_max = max(selfcheckgpt_scores)

        return {'avg-selfcheckgpt': selfcheckgpt_scores_avg, 'max-selfcheckgpt': selfcheckgpt_scores_max}

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {k: mean for k in ["avg-selfcheckgpt", "max-selfcheckgpt"]}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {k: False for k in ["avg-selfcheckgpt", "max-selfcheckgpt"]}
