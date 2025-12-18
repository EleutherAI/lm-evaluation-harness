import os
from dotenv import load_dotenv

import datasets

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    pass


LLM_JUDGE_SET_SIZE = os.getenv("LLM_JUDGE_SET_SIZE", 500)
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-5.1")


def load_dataset(**kwargs):
    dataset = datasets.load_dataset(
        "p-1-ai/nl2foam-mt-fields-ift", split="validation"
    ).select(range(LLM_JUDGE_SET_SIZE))

    return {"validation": dataset}


# LLM as a judge utils
def build_user_prompt(input: str, gt: str, pred: str) -> str:
    return (
f"Given the INPUT "
f"and CORRECT OUTPUT, "
f"score the MODEL PREDICTION"
f" on a scale from 0 to 100, where 100 is the best score. No explanation. No extra characters. Just a single score.\n\n"
f"INPUT:\n{input}\n"
f"CORRECT OUTPUT:\n{gt}\n"
f"MODEL PREDICTION:\n{pred}\n"
	)


def judge_eval(doc: dict, results: list[str]) -> dict[str, float]:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    user_prompt = build_user_prompt(doc["input"], doc["output"], results[0])

    resp = client.chat.completions.create(
        model=MODEL_VERSION,
        reasoning_effort="low", # temperature=0 not applicable for gpt-5.1
        max_completion_tokens=1024, # max_tokens=1024 not applicable for gpt-5.1
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = resp.choices[0].message.content.strip()

    return {
        "llm_judge": float(raw)
    }

