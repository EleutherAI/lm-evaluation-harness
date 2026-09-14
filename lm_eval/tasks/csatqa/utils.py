import datasets


def _process_doc(doc):
    instruction = f"""다음을 읽고 정답으로 알맞은 것을 고르시요.
### Context: {doc["context"]}
### Question: {doc["question"]}
### Options:
(1) {doc["option#1"]}\n(2) {doc["option#2"]}\n(3) {doc["option#3"]}\n(4) {doc["option#4"]}\n(5) {doc["option#5"]}
### Answer: 주어진 문제의 정답은"""

    out_doc = {
        "question": instruction,
        "choices": ["(1)", "(2)", "(3)", "(4)", "(5)"],
        "gold": int(doc["gold"]) - 1,
    }
    return out_doc


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_process_doc)


def _process_category(dataset: datasets.Dataset, category: str) -> datasets.Dataset:
    """Keep only one CSAT-QA category, then apply the shared prompt formatting.

    The dataset used to ship a loading script that exposed one builder config
    per category and filtered on the ``Category`` column itself. The script was
    removed (``datasets`` no longer runs them), so the data is now read straight
    from ``data/csatqa.json``, which holds every category in a single file. The
    per-category selection therefore has to happen here instead, otherwise every
    subject task would score all 936 rows.
    """
    return dataset.filter(lambda doc: doc["Category"] == category).map(_process_doc)


def process_docs_wr(dataset: datasets.Dataset) -> datasets.Dataset:
    return _process_category(dataset, "WR")


def process_docs_gr(dataset: datasets.Dataset) -> datasets.Dataset:
    return _process_category(dataset, "GR")


def process_docs_rcs(dataset: datasets.Dataset) -> datasets.Dataset:
    return _process_category(dataset, "RCS")


def process_docs_rcss(dataset: datasets.Dataset) -> datasets.Dataset:
    return _process_category(dataset, "RCSS")


def process_docs_rch(dataset: datasets.Dataset) -> datasets.Dataset:
    return _process_category(dataset, "RCH")


def process_docs_li(dataset: datasets.Dataset) -> datasets.Dataset:
    return _process_category(dataset, "LI")
