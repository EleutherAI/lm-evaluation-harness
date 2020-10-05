from . common import HFNLPTask
from ..utils_stream import X, each, apply, join, filt, one
import collections
import nlp


class RACE(HFNLPTask):
    NLP_PATH = "race"
    NLP_NAME = "high"

    cache = {}

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _collate_data(self, set):
        if set in self.cache: return self.cache[set]
        # One big issue with HF's implementation of this dataset: it makes a
        # separate document for each question; meanwhile, in the GPT3 paper it
        # is shown that one document is made per passage.

        r = collections.defaultdict(list)
        for item in nlp.load_dataset(path=self.NLP_PATH, name=self.NLP_NAME)[set]:
            r[item['article']].append(item)
        
        res = list(r.values() >> each(lambda x: {
            'article': x[0]['article'],
            'problems': x >> each(lambda y: {
                'question': y['question'],
                'answer': y['answer'],
                'options': y['options'],
            })
        }))

        self.cache[set] = res
        return res

    def training_docs(self):
        return self._collate_data("train")

    def validation_docs(self):
        return self._collate_data("validation")

    def test_docs(self):
        return self._collate_data("test")

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc, include_target=True):
        print(doc)
        r = "Article:\n" + doc['article'] + '\n\n'

        r += doc['problems'] >> each(
            lambda x: 'Q: ' + x['question'] + '\n\nA: ' + x['options'][['A', 'B', 'C', 'D'].index(x['answer'])]) \
                >> join('\n\n')

        return r

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        # TODO: implement
        raise NotImplementedError()