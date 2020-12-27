from . common import HFTask

class NaturalQs(HFTask):
    DATASET_PATH = "natural_questions"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def training_docs(self):
        # Cache training for faster few-shot.
        # Data is too large to fit in memory.
        return self.data["train"]

    def fewshot_examples(self, k):
        # Data is too large to fit in memory. We just sample from the first bit.
        if self._traindocs is None:
            self._traindocs = list(islice(self.training_docs(), 0, 100000))

        return random.sample(self._traindocs, k)

    def doc_to_text(self, doc, include_target=True):
        question = doc['question']['text']
        
        text = 'Q: ' + question + '\n\n' + 'A: '

        if include_target:
            # There's a short answer and a long answer. Based on the paper, I'm using the long answer.
            short_answer = doc['annotations']['short_answers'][0]['text']
            long_answer_start = doc['annotations']['long_answer'][0]['start_token']
            long_answer_end = doc['annotations']['long_answer'][0]['end_token']
            long_answer_span = doc['document']['tokens']['token'][long_answer_start:long_answer_end]
            long_answer_is_html = doc['document']['tokens']['is_html'][long_answer_start:long_answer_end]
            long_answer_chars = [tok for (tok, is_html) in zip(long_answer_span, long_answer_is_html) if not is_html]
            long_answer = " ".join(long_answer_chars)
            text += long_answer # Replace with short_answer[0] for short answer

        return text

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # TODO: implement
        raise NotImplementedError()