from lm_eval.tasks.code_eval_sandbox import pass_at_k as sandbox_pass_at_k


def pass_at_k(
    references: list[str],
    predictions: list[list[str]],
    k: int | list[int] | None = None,
):
    assert k is not None
    return sandbox_pass_at_k(references, predictions, k)


def canonical_solution(doc: dict) -> str:
    """Return the reference implementation rather than the test harness."""
    return doc["canonical_solution"]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs, strict=True)
    ]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs, strict=True)
    ]
