"""Document filter for the GSPC tasks.

Each GSPC bank holds one canary row, marked in its `_canary` field and
carrying no expected label. It is there to detect training-data
contamination and is never scored.
"""


def process_docs(dataset):
    return dataset.filter(
        lambda doc: doc.get("category") != "canary"
        and not doc.get("_canary")
        and str(doc.get("expected") or "").strip() != ""
    )
