import os
from dotenv import load_dotenv

import datasets

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    pass


LLM_JUDGE_SET_SIZE = 100
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4.1-mini")

SYSTEM_PROMPT = """You are an grader for response generation.
You are given:
1) the ground truth response
2) a predicted response

Your job:
- Compare the ground truth response and the predicted response.
- Output a float number between 0.0 and 1.0, the ratio of how similar the predicted response is to the ground truth response.
No explanation. No extra characters. Just a single float number between 0.0 and 1.0
"""


def load_dataset(**kwargs):
    dataset = datasets.load_dataset(
        "p-1-ai/nl2foam-mt-fields-ift", split="validation"
    ).select(range(LLM_JUDGE_SET_SIZE))

    return {"validation": dataset}


# LLM as a judge utils
def build_user_prompt(instruction: str, gt: str, pred: str) -> str:
    return f"""
Instruction:
{instruction.strip()}

Ground Truth:
{gt.strip()}

Predicted:
{pred.strip()}
"""


def judge_eval(doc: dict, results: list[str]) -> dict[str, float]:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    user_prompt = build_user_prompt(doc["instruction"], doc["output"], results[0])

    resp = client.chat.completions.create(
        model=MODEL_VERSION,
        temperature=0,
        max_tokens=10,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = resp.choices[0].message.content.strip()

    return {
        "llm_judge": float(raw)
    }

