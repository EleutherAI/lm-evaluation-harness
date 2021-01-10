# REMINDER: this code needs to be rewritten for the new framework. Remove this comment when the code is fully converted.

import numpy as np
from tqdm import auto as tqdm_lib
from . common import HFTask, simple_accuracy_metric, yesno, trueneitherfalse
from lm_eval.base import rf, mean

class BoolQ(HFTask):
    DATASET_PATH = "super_glue"
    DATASET_NAME = "boolq"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        # TODO: figure out actual description
        return "Read the following passages and answer each question with a yes or a no."

    def doc_to_text(self, doc):
        return f"{doc['passage']}\nquestion: {doc['question']}\nanswer: "
    
    def doc_to_target(self, doc):
        return yesno(doc['label']) 

    def construct_requests(self, doc, ctx):

        ll_yes, _ = rf.loglikelihood(ctx, ' yes')
        ll_no , _ = rf.loglikelihood(ctx, ' no')

        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]

        acc = 1. if (ll_yes > ll_no) == gold else 0.

        return {
            "acc": acc
        }
    
    def higher_is_better(self):
        return {
            "acc": True
        }
    
    def aggregation(self):
        return {
            "acc": mean
        }


class CommitmentBank(HFTask):
    DATASET_PATH = "super_glue"
    DATASET_NAME = "cb"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        return "Given a premise and a hypothesis, classify whether the author of the premise is committed to the truth of the hypothesis. The three possible labels are true, false or neither."

    def doc_to_text(self, doc, include_target=True):
        text = "{}\nquestion:\t{}\ttrue, false or neither?\nanswer:".format(
            doc["premise"],
            doc["hypothesis"],
        )
        if include_target:
            # True = entailment
            # False = contradiction
            # Neither = neutral
            text += " {}".format({0: "true", 1: "neither", 2: "false"}[doc["label"]])
        return text
    
    def doc_to_target(self, doc):
        return trueneitherfalse(doc['label']) 

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, ' true')
        ll_neither, _ = rf.loglikelihood(ctx, ' neither')
        ll_false, _ = rf.loglikelihood(ctx, ' false')

        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        gold = doc["label"]
        acc = 1. if (np.argmax(results)) == gold else 0.

        return {
            "acc": acc
        }
    
    def higher_is_better(self):
        return {
            "acc": True
        }
    
    def aggregation(self):
        return {
            "acc": mean
        }

class Copa(HFTask):
    DATASET_PATH = "super_glue"
    DATASET_NAME = "copa"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc, include_target=True):
        # Drop the period
        connector = {
            "cause": "because",
            "effect": "therefore",
        }[doc["question"]]
        text = doc["premise"].strip()[:-1] + f" {connector} "
        if include_target:
            correct_choice = doc["choice1"] if doc["label"] == 0 else doc["choice2"]
            # Connect the sentences
            text += self.convert_choice(correct_choice)
        return text

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # TODO: Implement evaluation code using new framework

        # ***IMPORTANT***: this evaluation function needs to be rewritten for the new framework. 
        # For more info, check out the interface in base.py and the example BoolQ implementation in superglue.py. 
        # Remove this comment when the evaluation code is implemented.
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in tqdm_lib.tqdm(docs):
            ctx = self.fewshot_context(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
            )
            choice1 = " " + self.convert_choice(doc["choice1"])
            choice2 = " " + self.convert_choice(doc["choice2"])
            preds.append(lm.loglikelihood(ctx, choice2) > lm.loglikelihood(ctx, choice1))
        return simple_accuracy_metric(preds=preds, golds=golds)

    @staticmethod
    def convert_choice(choice):
        return choice[0].lower() + choice[1:]


class MultiRC(HFTask):
    DATASET_PATH = "super_glue"
    DATASET_NAME = "multirc"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        return "READING COMPREHENSION ANSWER KEY"

    def doc_to_text(self, doc, include_target=True):
        return f"{doc['paragraph']}\n\n{doc['question']}\n" \
            + (self.format_answer(answer=doc["answer"], label=doc["label"])
               if include_target else "")

    @staticmethod
    def format_answer(answer, label):
        label_str = "True" if label else "False"
        return f"[{label_str}] {answer}"

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # TODO: Implement evaluation code using new framework

        # ***IMPORTANT***: this evaluation function needs to be rewritten for the new framework. 
        # For more info, check out the interface in base.py and the example BoolQ implementation in superglue.py. 
        # Remove this comment when the evaluation code is implemented.
        preds = []
        for doc in docs:
            ctx = self.fewshot_context(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
            )
            true_choice = self.format_answer(answer=doc["answer"], label=True)
            false_choice = self.format_answer(answer=doc["answer"], label=False)
            preds.append(
                lm.loglikelihood(ctx, f' {true_choice}')
                > lm.loglikelihood(ctx, f' {false_choice}')
            )

        # Only count as correct if all answers are labeled correctly for each question
        question_scoring_dict = {}
        for doc, pred in zip(docs, preds):
            question_id = doc["idx"]["question"]
            if question_id not in question_scoring_dict:
                question_scoring_dict[question_id] = []
            gold_label = doc["label"] == 1
            question_scoring_dict[question_id].append(gold_label == pred)
        acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
        return {
            "major": acc,
            "minor": {"acc": acc},
            "higher_is_better": True,
        }


class WordsInContext(HFTask):
    DATASET_PATH = "super_glue"
    DATASET_NAME = "wic"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc, include_target=True):
        text = "{}\n{}\nquestion\tIs the word '{}' used in the same way in the" \
               " two sentences above?\nanswer:".format(
                    doc["sentence1"],
                    doc["sentence2"],
                    doc["sentence1"][doc["start1"]:doc["end1"]],
                )
        if include_target:
            text += " {}".format({0: "no", 1: "yes"}[doc["label"]])
        return text

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # TODO: Implement evaluation code using new framework

        # ***IMPORTANT***: this evaluation function needs to be rewritten for the new framework. 
        # For more info, check out the interface in base.py and the example BoolQ implementation in superglue.py. 
        # Remove this comment when the evaluation code is implemented.
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in tqdm_lib.tqdm(docs):
            ctx = self.fewshot_context(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
            )
            preds.append(lm.loglikelihood(ctx, ' yes') > lm.loglikelihood(ctx, ' no'))
        return simple_accuracy_metric(preds=preds, golds=golds)


class SGWinogradSchemaChallenge(HFTask):
    DATASET_PATH = "super_glue"
    DATASET_NAME = "wsc"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                # GPT-3 Paper's format only uses positive examples
                self._training_docs = [
                    doc for doc in
                    self._load_nlp_dataset()["train"]
                    if doc["label"]
                ]
            return self._training_docs

    def fewshot_description(self):
        return "Final Exam with Answer Key\n" \
           "Instructions: Please carefully read the following passages. " \
           "For each passage, you must identify which noun the pronoun marked in *bold*" \
           " refers to.\n====="

    def doc_to_text(self, doc, include_target=True):
        raw_passage = doc["text"]
        passage = (
            raw_passage[:doc["span2_index"]]
            + "*{}*".format(doc["span2_text"])
            + raw_passage[doc["span2_index"] + len(doc["span2_text"]):]
        )
        pronoun = doc["span2_text"]
        text = (
            f"Passage: {passage}\n"
            + f"Question: In the passage above, what does the pronoun \"*{pronoun}*\" refer to?\n"
            + "Answer:"
        )
        if include_target:
            text += " {}".format(doc["span1_text"])
        return text

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # TODO: Implement evaluation code using new framework

        # ***IMPORTANT***: this evaluation function needs to be rewritten for the new framework. 
        # For more info, check out the interface in base.py and the example BoolQ implementation in superglue.py. 
        # Remove this comment when the evaluation code is implemented.
        golds = [doc["label"] for doc in docs]
        preds = []
        for doc in tqdm_lib.tqdm(docs):
            ctx = self.fewshot_context(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
            )
            to_predict = " " + doc["span1_text"]
            num_tokens = len(lm.tokenizer.tokenize(to_predict))
            generated = lm.generate(
                context=ctx,
                max_gen_length=num_tokens,
            )
            preds.append(1 if generated == to_predict else 0)
        return simple_accuracy_metric(preds=preds, golds=golds)

class RTE(HFTask):
    DATASET_PATH = "super_glue"
    DATASET_NAME = "rte"

    def fewshot_description(self):
        #TODO: implement
        pass

    def doc_to_text(self, doc, include_target=True):
        if include_target:
            if doc['label'] == 0:
                answer = 'True'
            else:
                answer = 'False'
            return ''.join([doc['premise'], '\nquestion: ',doc['hypothesis'], ' True or False?\nanswer: ', answer])
        else:
            return ''.join([doc['premise'], '\nquestion: ',doc['hypothesis'], ' True or False?\nanswer: '])
    
    # TODO: Implement evaluation code

    # ***IMPORTANT***: this evaluation function needs to be written for the new framework. 
    # For more info, check out the interface in base.py and the example BoolQ implementation in superglue.py. 
    # Remove this comment when the evaluation code is implemented.

