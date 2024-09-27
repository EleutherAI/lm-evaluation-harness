import random
from typing import List

import numpy as np
import pytest

from lm_eval import tasks
from lm_eval.tasks import TaskManager
from lm_eval.utils import join_iters


MMLU_ANATOMY_ZERO_SHOT = """The following are multiple choice questions (with answers) about anatomy.

A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral
A. paralysis of the facial muscles.
B. paralysis of the facial muscles and loss of taste.
C. paralysis of the facial muscles, loss of taste and lacrimation.
D. paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.
Answer:"""

MMLU_ANATOMY_FIVE_SHOT = """The following are multiple choice questions (with answers) about anatomy.

What is the embryological origin of the hyoid bone?
A. The first pharyngeal arch
B. The first and second pharyngeal arches
C. The second pharyngeal arch
D. The second and third pharyngeal arches
Answer: D

Which of these branches of the trigeminal nerve contain somatic motor processes?
A. The supraorbital nerve
B. The infraorbital nerve
C. The mental nerve
D. None of the above
Answer: D

The pleura
A. have no sensory innervation.
B. are separated by a 2 mm space.
C. extend into the neck.
D. are composed of respiratory epithelium.
Answer: C

In Angle's Class II Div 2 occlusion there is
A. excess overbite of the upper lateral incisors.
B. negative overjet of the upper central incisors.
C. excess overjet of the upper lateral incisors.
D. excess overjet of the upper central incisors.
Answer: C

Which of the following is the body cavity that contains the pituitary gland?
A. Abdominal
B. Cranial
C. Pleural
D. Spinal
Answer: B

A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral
A. paralysis of the facial muscles.
B. paralysis of the facial muscles and loss of taste.
C. paralysis of the facial muscles, loss of taste and lacrimation.
D. paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.
Answer:"""


@pytest.mark.parametrize(
    "task_names,sets,num_fewshot,seed,num_examples,expected_prompt",
    [
        (["mmlu_anatomy"], "test", 0, 42, 1, MMLU_ANATOMY_ZERO_SHOT),
        (["mmlu_anatomy"], "test", 5, 42, 1, MMLU_ANATOMY_FIVE_SHOT),
    ],
)
def test_mmlu_prompt_rendering(
    task_names: List[str],
    sets: str,
    num_fewshot: int,
    seed: int,
    num_examples: int,
    expected_prompt: str,
):
    np.random.seed(seed)

    task_manager = TaskManager()
    task_dict = tasks.get_task_dict(task_names, task_manager)

    for task_name, task in task_dict.items():
        if isinstance(task, tuple):
            _, task = task

        rnd = random.Random()
        rnd.seed(seed)

        iters = []

        for set in sets.split(","):
            docs = None
            if set == "train" and task.has_training_docs():
                docs = task.training_docs()
            if set == "val" and task.has_validation_docs():
                docs = task.validation_docs()
            if set == "test" and task.has_test_docs():
                docs = task.test_docs()
            if docs is not None:
                iters.append(docs)

        if len(iters) == 0:
            raise ValueError

        docs = join_iters(iters)

        for i, doc in (
            zip(range(num_examples), docs) if num_examples > 0 else enumerate(docs)
        ):
            ctx = task.fewshot_context(
                doc=doc,
                num_fewshot=num_fewshot,
            )

            assert ctx == expected_prompt
