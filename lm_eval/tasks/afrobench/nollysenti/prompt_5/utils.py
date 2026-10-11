from lm_eval.utils import weighted_f1_score  # noqa


def process_docs(dataset):
    # The Hub dataset dropped its loading script; the TSVs name the text
    # column after the language (en_review, ha_review, ...) and the label
    # column `sentiment`.
    lang = next(c for c in dataset.column_names if c.endswith("_review"))
    return dataset.rename_columns({lang: "review", "sentiment": "label"})
