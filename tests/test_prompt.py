import random

import numpy as np
import pytest

from lm_eval.api.instance import Instance
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

# Expected request arguments for construct_requests
# Each tuple is (context, continuation) for loglikelihood request
# MMLU uses doc_to_choice: ["A", "B", "C", "D"] and target_delimiter: " "
MMLU_ANATOMY_ZERO_SHOT_REQUESTS = [
    (MMLU_ANATOMY_ZERO_SHOT, " A"),
    (MMLU_ANATOMY_ZERO_SHOT, " B"),
    (MMLU_ANATOMY_ZERO_SHOT, " C"),
    (MMLU_ANATOMY_ZERO_SHOT, " D"),
]

MMLU_ANATOMY_FIVE_SHOT_REQUESTS = [
    (MMLU_ANATOMY_FIVE_SHOT, " A"),
    (MMLU_ANATOMY_FIVE_SHOT, " B"),
    (MMLU_ANATOMY_FIVE_SHOT, " C"),
    (MMLU_ANATOMY_FIVE_SHOT, " D"),
]

# Expected prompts with gen_prefix="The answer is:"
# Fewshot answers get gen_prefix prepended, target question ends with gen_prefix
MMLU_ANATOMY_ZERO_SHOT_WITH_GEN_PREFIX = """The following are multiple choice questions (with answers) about anatomy.

A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral
A. paralysis of the facial muscles.
B. paralysis of the facial muscles and loss of taste.
C. paralysis of the facial muscles, loss of taste and lacrimation.
D. paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.
Answer: The answer is:"""

MMLU_ANATOMY_FIVE_SHOT_WITH_GEN_PREFIX = """The following are multiple choice questions (with answers) about anatomy.

What is the embryological origin of the hyoid bone?
A. The first pharyngeal arch
B. The first and second pharyngeal arches
C. The second pharyngeal arch
D. The second and third pharyngeal arches
Answer: The answer is: D

Which of these branches of the trigeminal nerve contain somatic motor processes?
A. The supraorbital nerve
B. The infraorbital nerve
C. The mental nerve
D. None of the above
Answer: The answer is: D

The pleura
A. have no sensory innervation.
B. are separated by a 2 mm space.
C. extend into the neck.
D. are composed of respiratory epithelium.
Answer: The answer is: C

In Angle's Class II Div 2 occlusion there is
A. excess overbite of the upper lateral incisors.
B. negative overjet of the upper central incisors.
C. excess overjet of the upper lateral incisors.
D. excess overjet of the upper central incisors.
Answer: The answer is: C

Which of the following is the body cavity that contains the pituitary gland?
A. Abdominal
B. Cranial
C. Pleural
D. Spinal
Answer: The answer is: B

A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral
A. paralysis of the facial muscles.
B. paralysis of the facial muscles and loss of taste.
C. paralysis of the facial muscles, loss of taste and lacrimation.
D. paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.
Answer: The answer is:"""

MMLU_ANATOMY_ZERO_SHOT_WITH_GEN_PREFIX_REQUESTS = [
    (MMLU_ANATOMY_ZERO_SHOT_WITH_GEN_PREFIX, " A"),
    (MMLU_ANATOMY_ZERO_SHOT_WITH_GEN_PREFIX, " B"),
    (MMLU_ANATOMY_ZERO_SHOT_WITH_GEN_PREFIX, " C"),
    (MMLU_ANATOMY_ZERO_SHOT_WITH_GEN_PREFIX, " D"),
]

MMLU_ANATOMY_FIVE_SHOT_WITH_GEN_PREFIX_REQUESTS = [
    (MMLU_ANATOMY_FIVE_SHOT_WITH_GEN_PREFIX, " A"),
    (MMLU_ANATOMY_FIVE_SHOT_WITH_GEN_PREFIX, " B"),
    (MMLU_ANATOMY_FIVE_SHOT_WITH_GEN_PREFIX, " C"),
    (MMLU_ANATOMY_FIVE_SHOT_WITH_GEN_PREFIX, " D"),
]


@pytest.mark.parametrize(
    "task_names,sets,num_fewshot,seed,num_examples,expected_prompt,expected_requests,gen_prefix",
    [
        (
            ["mmlu_anatomy"],
            "test",
            0,
            42,
            1,
            MMLU_ANATOMY_ZERO_SHOT,
            MMLU_ANATOMY_ZERO_SHOT_REQUESTS,
            None,
        ),
        (
            ["mmlu_anatomy"],
            "test",
            5,
            42,
            1,
            MMLU_ANATOMY_FIVE_SHOT,
            MMLU_ANATOMY_FIVE_SHOT_REQUESTS,
            None,
        ),
        (
            ["mmlu_anatomy"],
            "test",
            0,
            42,
            1,
            MMLU_ANATOMY_ZERO_SHOT_WITH_GEN_PREFIX,
            MMLU_ANATOMY_ZERO_SHOT_WITH_GEN_PREFIX_REQUESTS,
            "The answer is:",
        ),
        (
            ["mmlu_anatomy"],
            "test",
            5,
            42,
            1,
            MMLU_ANATOMY_FIVE_SHOT_WITH_GEN_PREFIX,
            MMLU_ANATOMY_FIVE_SHOT_WITH_GEN_PREFIX_REQUESTS,
            "The answer is:",
        ),
    ],
)
def test_mmlu_prompt_rendering(
    task_names: list[str],
    sets: str,
    num_fewshot: int,
    seed: int,
    num_examples: int,
    expected_prompt: str,
    expected_requests: list[tuple[str, str]],
    gen_prefix: str | None,
):
    np.random.seed(seed)

    task_manager = TaskManager()
    task_dict = task_manager.load(task_names)["tasks"]

    for _task_name, task in task_dict.items():
        # Apply gen_prefix to task config if provided
        if gen_prefix is not None:
            task.config.gen_prefix = gen_prefix
            task.fewshot_cfg.gen_prefix = gen_prefix

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

        for _, doc in (
            zip(range(num_examples), docs, strict=False)
            if num_examples > 0
            else enumerate(docs)
        ):
            ctx = task.fewshot_context(
                doc=doc,
                num_fewshot=num_fewshot,
                gen_prefix=gen_prefix,
            )

            assert ctx == expected_prompt

            # Test construct_requests
            requests = task.construct_requests(doc=doc, ctx=ctx)

            # MMLU is multiple_choice, so we expect a list of Instance objects
            assert isinstance(requests, list), (
                "construct_requests should return a list for multiple_choice tasks"
            )
            assert len(requests) == 4, (
                "MMLU should have 4 requests (one per choice A, B, C, D)"
            )

            for i, req in enumerate(requests):
                assert isinstance(req, Instance), f"Request {i} should be an Instance"
                assert req.request_type == "loglikelihood", (
                    f"Request {i} should be loglikelihood type"
                )
                assert req.idx == i, f"Request {i} should have idx={i}"
                assert req.arguments == expected_requests[i], (
                    f"Request {i} arguments mismatch.\n"
                    f"Expected: {expected_requests[i]}\n"
                    f"Got: {req.arguments}"
                )
