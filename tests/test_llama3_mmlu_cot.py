from pathlib import Path

import pytest

from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.utils import apply_template


TEMPLATE = (
    Path(__file__).parents[1]
    / "lm_eval"
    / "tasks"
    / "llama3"
    / "instruct"
    / "mmlu_cot"
    / "_mmlu_cot_llama_template_yaml"
)

# User turns copied from `input_final_prompts` in Meta's Llama-3.1-8B-Instruct
# MMLU (0-shot, CoT) eval details. Meta keeps the question's surrounding
# whitespace, which some MMLU questions have.
META_HEAD = (
    "Given the following question and four candidate answers (A, B, C and D), "
    "choose the best answer.\n\nQuestion: "
)
META_TAIL = (
    "\n\n- For simple problems:\nDirectly provide the answer with minimal "
    "explanation.\n\n- For complex problems:\nUse this step-by-step format:\n"
    "## Step 1: [Concise description]\n[Brief explanation]\n"
    "## Step 2: [Concise description]\n[Brief explanation]\n\n"
    "Regardless of the approach, always conclude with:\n"
    "The best answer is [the_answer_letter].\n"
    "where the [the_answer_letter] is one of A, B, C or D.\n\n"
    "Let's think step by step."
)


@pytest.mark.parametrize(
    "doc, expected",
    [
        (
            {
                "question": " How many books are in the New Testament?",
                "choices": ["30", "29", "27", "47"],
            },
            META_HEAD
            + " How many books are in the New Testament?\nA. 30\nB. 29\nC. 27\nD. 47"
            + META_TAIL,
        ),
        (
            {
                "question": "What are histones?\n",
                "choices": ["Lipids", "Carbohydrates", "Nucleotides", "Proteins"],
            },
            META_HEAD
            + "What are histones?\n\nA. Lipids\nB. Carbohydrates\nC. Nucleotides\nD. Proteins"
            + META_TAIL,
        ),
    ],
    ids=["leading_space", "trailing_newline"],
)
def test_mmlu_cot_llama_prompt_matches_meta(doc, expected):
    config = load_yaml(TEMPLATE, resolve_func=False)
    assert apply_template(config["doc_to_text"], doc) == expected
