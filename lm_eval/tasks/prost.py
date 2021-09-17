"""
PROST: Physical Reasoning about Objects Through Space and Time
https://arxiv.org/pdf/2106.03634.pdf

NOTE: PROST is limited to the zero-shot setting to adhere to authors' intentions
as discussed in section 7 of the paper: "We hope that the community will use
this dataset in the intended way: in a zero-shot setting to probe models which
have been trained on data not specifically collected to succeed on PROST."

# TODO: Update citation when it is made available at https://github.com/nala-cub/prost.
@misc{arocaouellette2021prost,
      title={PROST: Physical Reasoning of Objects through Space and Time}, 
      author={Stéphane Aroca-Ouellette and Cory Paik and Alessandro Roncone and Katharina Kann},
      year={2021},
      eprint={2106.03634},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

from lm_eval.base import MultipleChoiceTask
from . common import HFTask


class PROST(HFTask, MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "corypaik/prost"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def fewshot_context(self, doc, num_fewshot, provide_description, rnd):
        assert num_fewshot == 0, 'PROST is designed to probe models in a zero-shot fashion only.'
        return super().fewshot_context(doc, num_fewshot, provide_description, rnd)

    def _convert_standard(self, doc):
        out_doc = {
            "query": f"{doc['context']}\nQuestion: {doc['ex_question']}\nAnswer:",
            "choices": [doc['A'], doc['B'], doc['C'], doc['D']],
            "gold": doc['label'],
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]
