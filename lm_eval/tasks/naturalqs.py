from . common import HFTask
import apache_beam

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

    def doc_to_text(self, doc, include_target=True):
        question = doc['question']['text']
        short_answer = doc['annotations']['short_answers'][0]['text']
        long_answer_start = doc['annotations']['long_answer'][0]['start_token']
        long_answer_end = doc['annotations']['long_answer'][0]['end_token']
        passage = " ".join(doc['document']['tokens']['token'][long_answer_start:long_answer_end])
        
        text = 'Q: ' + question + '\n\n' + 'A: '

        if include_target:
            # What if there is no short answer? This will be an empty string. Currently, default to the long answer otherwise.
            if short_answer:
                text += short_answer[0]
            else:
                text += long_answer

        return text

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # TODO: implement
        raise NotImplementedError()