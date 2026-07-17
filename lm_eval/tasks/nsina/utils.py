import datasets


CATEGORY_LABELS = ["Business", "International News", "Local News", "Sports"]

# Raw `Source` values as they appear in the dataset, index-aligned with the
# doc_to_choice list in nsina_media.yaml.
MEDIA_LABELS = [
    "Adaderana",
    "ITN news",
    "Lankatruth",
    "Siyatha News",
    "dinamina",
    "divaina",
    "hirunews",
    "https://sinhala.news.lk",
    "www.lankadeepa.lk",
    "www.vikalpa.org",
]


def _truncate(text: str, max_chars: int = 2000) -> str:
    """Truncate long articles to keep prompts within reasonable context."""
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text


def process_docs_categories(dataset: datasets.Dataset) -> datasets.Dataset:
    # The reference implementation drops rows with missing values:
    # https://github.com/Sinhala-NLP/Sinhala-News-Category-Prediction
    dataset = dataset.filter(
        lambda doc: doc["News Content"] is not None and doc["Category"] is not None
    )

    def _process_doc(doc):
        return {
            "text": _truncate(doc["News Content"]),
            "label": CATEGORY_LABELS.index(doc["Category"]),
        }

    return dataset.map(_process_doc)


def process_docs_media(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(
        lambda doc: doc["News Content"] is not None and doc["Source"] is not None
    )

    def _process_doc(doc):
        return {
            "text": _truncate(doc["News Content"]),
            "label": MEDIA_LABELS.index(doc["Source"]),
        }

    return dataset.map(_process_doc)


def process_docs_headlines(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(
        lambda doc: doc["News Content"] is not None and doc["Headline"] is not None
    )

    def _process_doc(doc):
        return {
            "text": _truncate(doc["News Content"]),
            "headline": doc["Headline"].strip(),
        }

    return dataset.map(_process_doc)


def macro_f1_score(items):
    """Macro-averaged F1 over (gold, prediction) index pairs, dependency-free."""
    golds, preds = zip(*items, strict=True)
    scores = []
    for label in set(golds):
        pairs = list(zip(golds, preds, strict=True))
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(scores) / len(scores)
