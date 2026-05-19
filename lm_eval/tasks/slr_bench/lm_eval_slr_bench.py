import shutil
import sys

from evaluate import load


if shutil.which("swipl") is None:
    sys.exit(
        "Error: SWI-Prolog (swipl) is not installed or not in PATH. Please install SWI-Prolog to use this task."
    )

try:
    symbolic_judge = load("AIML-TUDA/VerifiableRewardsForScalableLogicalReasoning")
except Exception as e:
    print(f"Warning: Could not load VerifiableRewards: {e}")
    symbolic_judge = None


def process_results(doc, results):
    """
    Process results for the SLR-Bench task.

    Args:
        doc: Document with ground truth and validation program
        results: Model output (generated text)

    Returns:
        Dictionary with metrics
    """

    prediction = results[0]

    # Create the reference in the required format
    try:
        reference = [
            {
                "validation_program": doc.get("validation program", ""),
                "evaluation_config": {
                    "positive_predicate": "eastbound",
                    "negative_predicate": "westbound",
                },
            }
        ]

        # Use symbolic judge if available
        if symbolic_judge is not None:
            results = symbolic_judge.compute(
                predictions=[prediction], references=reference
            )

            if isinstance(results, dict) and "accuracy" in results:
                return {"verifiable_reward": results["accuracy"]}

        # Fallback: exact match
        target = doc.get("ground-truth rule", "")
        exact_match = float(prediction.strip() == target.strip())
        return {"verifiable_reward": exact_match}

    except Exception as e:
        print(f"Error in process_results: {e}")
        return {"verifiable_reward": 0.0}
