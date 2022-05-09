"""
The LAMBADA dataset: Word prediction requiring a broad discourse context∗
https://arxiv.org/pdf/1606.06031.pdf

The LAMBADA dataset machine-translated to other languages.
LAMBADA is a dataset to evaluate the capabilities of computational models for text
understanding by means of a word prediction task. LAMBADA is a collection of narrative
passages sharing the characteristic that human subjects are able to guess their last
word if they are exposed to the whole passage, but not if they only see the last
sentence preceding the target word. To succeed on LAMBADA, computational models
cannot simply rely on local context, but must be able to keep track of information
in the broader discourse.

Homepage: https://zenodo.org/record/2630551#.X4Xzn5NKjUI
"""
from . import lambada


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


class MultilingualLAMBADA(lambada.LAMBADA):
    VERSION = 0


class MultilingualLAMBADAEN(MultilingualLAMBADA):
    DATASET_NAME = 'en'


class MultilingualLAMBADAFR(MultilingualLAMBADA):
    DATASET_NAME = 'fr'


class MultilingualLAMBADADE(MultilingualLAMBADA):
    DATASET_NAME = 'de'


class MultilingualLAMBADAIT(MultilingualLAMBADA):
    DATASET_NAME = 'it'


class MultilingualLAMBADAES(MultilingualLAMBADA):
    DATASET_NAME = 'es'


LANG_CLASSES = [MultilingualLAMBADAEN, MultilingualLAMBADAFR,
                MultilingualLAMBADADE, MultilingualLAMBADAIT,
                MultilingualLAMBADAES]


def construct_tasks():
    tasks = {}
    for lang_class in LANG_CLASSES:
        tasks[f"lambada_mt_{lang_class.DATASET_NAME}"] = lang_class
    return tasks
