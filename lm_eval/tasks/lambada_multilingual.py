"""
The LAMBADA (OpenAI) dataset: Word prediction requiring a broad discourse context∗
https://arxiv.org/pdf/1606.06031.pdf

The LAMBADA OpenAI dataset machine-translated to other languages.
LAMBADA is a dataset to evaluate the capabilities of computational models for text
understanding by means of a word prediction task. LAMBADA is a collection of narrative
passages sharing the characteristic that human subjects are able to guess their last
word if they are exposed to the whole passage, but not if they only see the last
sentence preceding the target word. To succeed on LAMBADA, computational models
cannot simply rely on local context, but must be able to keep track of information
in the broader discourse.

Homepage: https://zenodo.org/record/2630551#.X4Xzn5NKjUI

Reference (OpenAI): https://github.com/openai/gpt-2/issues/131#issuecomment-497136199
"""
import inspect
from .lambada import LambadaOpenAI
from lm_eval.base import rf
import lm_eval.datasets.lambada_ja.lambada_ja
from lm_eval.metrics import mean, perplexity

_CITATION = """
@misc{
    author={Paperno, Denis and Kruszewski, Germán and Lazaridou, Angeliki and Pham, Quan Ngoc and Bernardi, Raffaella and Pezzelle, Sandro and Baroni, Marco and Boleda, Gemma and Fernández, Raquel},
    title={The LAMBADA dataset},
    DOI={10.5281/zenodo.2630551},
    publisher={Zenodo},
    year={2016},
    month={Aug}
}
"""


class LambadaOpenAIMultilingualEnglish(LambadaOpenAI):
    VERSION = 0
    DATASET_NAME = "en"


class LambadaOpenAIMultilingualFrench(LambadaOpenAI):
    VERSION = 0
    DATASET_NAME = "fr"


class LambadaOpenAIMultilingualGerman(LambadaOpenAI):
    VERSION = 0
    DATASET_NAME = "de"


class LambadaOpenAIMultilingualItalian(LambadaOpenAI):
    VERSION = 0
    DATASET_NAME = "it"


class LambadaOpenAIMultilingualSpanish(LambadaOpenAI):
    VERSION = 0
    DATASET_NAME = "es"


class LambadaOpenAIMultilingualJapanese(LambadaOpenAI):
    VERSION = 0
    DATASET_PATH = inspect.getfile(lm_eval.datasets.lambada_ja.lambada_ja)
    DATASET_NAME = "ja"


    def test_docs(self):
        # TODO: because all lambda texts are not translated yet, only take 1k translated texts
        # return self.dataset['test']
        texts = [item['text'] for item in self.dataset['test'] if item['text'] != ''][:1000]
        # remove last 。
        texts = [text[:-1] if text[-1] == "。" else text for text in texts]
        return texts
        # for doc in self.dataset["test"]:
        #     yield doc["text"]

    def doc_to_text(self, doc):
        # return doc.rsplit(" ", 1)[0]
        return doc
        # # using janome
        # try:
        #     from janome.tokenizer import Tokenizer
        #     t = Tokenizer()
        # except ImportError:
        #     raise ImportError("Please install janome first! (`pip install janome`)")
        # words = [token.surface for token in t.tokenize(doc)][:-1]
        # return "".join(words)

    def doc_to_target(self, doc):
        # return " " + doc["text"].rsplit(" ", 1)[1]
        # # take last token from context
        return "__lasttoken__"
        # # using janome
        # try:
        #     from janome.tokenizer import Tokenizer
        #     t = Tokenizer()
        # except ImportError:
        #     raise ImportError("Please install janome first! (`pip install janome`)")
        # word = [token.surface for token in t.tokenize(doc)][-1]
        # return word
    
    def construct_requests(self, doc, ctx):
        ll, is_greedy = rf.loglikelihood(ctx, self.doc_to_target(doc))

        return ll, is_greedy


LANG_CLASSES = [
    LambadaOpenAIMultilingualEnglish,
    LambadaOpenAIMultilingualFrench,
    LambadaOpenAIMultilingualGerman,
    LambadaOpenAIMultilingualItalian,
    LambadaOpenAIMultilingualSpanish,
    LambadaOpenAIMultilingualJapanese,
]


def construct_tasks():
    tasks = {}
    for lang_class in LANG_CLASSES:
        tasks[f"lambada_openai_mt_{lang_class.DATASET_NAME}"] = lang_class
    return tasks
