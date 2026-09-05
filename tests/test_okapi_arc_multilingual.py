import datasets

from lm_eval.tasks.okapi.arc_multilingual.utils import process_docs


def test_process_docs_preserves_arc_content_verbatim():
    dataset = datasets.Dataset.from_list(
        [
            {
                "id": "example",
                "instruction": "Allele [E] is dominant over [e]; parent [Ee] is heterozygous.",
                "option_a": "[EE]",
                "option_b": "[Ee]",
                "option_c": "[ee]",
                "option_d": "none",
                "option_e": None,
                "answer": "B",
            }
        ]
    )

    doc = process_docs(dataset)[0]

    assert doc["query"] == (
        "Question: Allele [E] is dominant over [e]; parent [Ee] is heterozygous.\n"
        "Answer:"
    )
    assert doc["choices"] == ["[EE]", "[Ee]", "[ee]", "none"]
    assert doc["gold"] == 1
