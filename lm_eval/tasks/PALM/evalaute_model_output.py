"""
This script is used to evaluate models' output using the LangChain evaluation
framework, which is the same pipeline used in the benchmark paper:

NOTE:
In the paper, they used the average scores of three LLMs: GPT-4, Qwen2.5-72B,
and CMDR+. This script uses only GPT-4.
"""

import os
import json
from tqdm import tqdm
from pathlib import Path
try:
    from langchain_openai import ChatOpenAI
    from langchain.evaluation import load_evaluator, Criteria
except ImportError:
    print(
        "The 'langchain' or 'langchain_openai' libraries are not installed.\n" +
        "Please install them using `pip install langchain langchain_openai`."
    )
    exit(1)


def parse_samples_json(samples_json_filepath):
    """
    Parses the samples JSON file and returns a list of dictionaries.
    """
    pass


def gpt4_as_a_judge(inputs, hyps, refs, criterion):
    # key OpenAI API token
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # load the judge model
    gpt4_judge = ChatOpenAI(
        model="gpt-4",
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
    )
    # Load evaluator
    evaluator = load_evaluator(
        evaluator="labeled_score_string",
        llm=gpt4_judge,
        criteria=["correctness", "coherence", "helpfulness", "detail"]
    )
    output_scores = []
    for input, hyp, ref in tqdm(zip(inputs, hyps, refs), total=len(inputs)):
        eval_result = evaluator.evaluate_strings(
            prediction=hyp,
            reference=ref,
            input=input,
        )
        output_scores.append({
            "input": input,
            "hypothesis": hyp,
            "reference": ref,
            "criterion": criterion.name,
            "score": eval_result["score"]
        })

    return output_scores


if __name__ == "__main__":
    samples_json_filepath = Path("samples.json")
    model_base_path = samples_json_filepath.parent
    timestamp = samples_json_filepath.stem.split("_")[-1]

    inputs, hyps, refs = parse_samples_json(samples_json_filepath)

    # same criteria as in the paper
    criteria = [
        Criteria.CORRECTNESS,
        Criteria.COHERENCE,
        Criteria.HELPFULNESS,
        Criteria.DETAIL
    ]

    for criterion in criteria:
        scores = gpt4_as_a_judge(inputs, hyps, refs, criterion)
        out_filepath = model_base_path / f"{criterion.name}-scores_{timestamp}.json"
        # write scores as JSONL file
        with open(out_filepath, "w") as f:
            for score in scores:
                f.write(json.dumps(score) + "\n")
        break # ONLY CALCULATE CORRECTNESS SCORES FOR NOW