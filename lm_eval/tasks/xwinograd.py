"""
It's All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning
https://arxiv.org/abs/2106.12066

Multilingual winograd schema challenge that includes English, French, Japanese, Portuguese, Russian and Chinese. Winograd schema challenges come from the XWinograd dataset introduced in Tikhonov et al. As it only contains 16 Chinese schemas, we add 488 Chinese schemas from clue/cluewsc2020.

Homepage: https://huggingface.co/datasets/Muennighoff/xwinograd
"""
from .winogrande import Winogrande


_CITATION = """
@misc{muennighoff2022crosslingual,
      title={Crosslingual Generalization through Multitask Finetuning},
      author={Niklas Muennighoff and Thomas Wang and Lintang Sutawika and Adam Roberts and Stella Biderman and Teven Le Scao and M Saiful Bari and Sheng Shen and Zheng-Xin Yong and Hailey Schoelkopf and Xiangru Tang and Dragomir Radev and Alham Fikri Aji and Khalid Almubarak and Samuel Albanie and Zaid Alyafeai and Albert Webson and Edward Raff and Colin Raffel},
      year={2022},
      eprint={2211.01786},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{tikhonov2021heads,
    title={It's All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning},
    author={Alexey Tikhonov and Max Ryabinin},
    year={2021},
    eprint={2106.12066},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_LANG = ["en", "fr", "jp", "pt", "ru", "zh"]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
    """
    return {f"xwinograd_{lang}": create_task(lang) for lang in _LANG}


def create_task(lang):
    class XWinograd(Winogrande):
        DATASET_PATH = "Muennighoff/xwinograd"
        DATASET_NAME = lang

        def __init__(self):
            super().__init__()

        def has_training_docs(self):
            return False

        def has_validation_docs(self):
            return False

        def has_test_docs(self):
            return True

        def training_docs(self):
            pass

        def validation_docs(self):
            pass

        def test_docs(self):
            return self.dataset["test"]

    return XWinograd
