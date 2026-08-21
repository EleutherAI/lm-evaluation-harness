from lm_eval.tasks.clinical_hallucination.preprocess_clinical import doc_to_target
from lm_eval.tasks.clinical_hallucination.metrics import hallucination_rate, hallucination_rate_per_sample


def test_clinical_hallucination_functions():
    doc = {
        "sent1": "A 45-year-old man presents with chest pain radiating to the left arm.",
        "sent2": "ECG shows ST elevation in leads II, III, and aVF.",
        "ending0": "Aortic dissection",
        "ending1": "Inferior ST-elevation myocardial infarction (STEMI)",
        "ending2": "Gastroesophageal reflux disease",
        "ending3": "Pericarditis",
        "label": 1,
    }

    # Test doc_to_target
    target = doc_to_target(doc)
    assert "Inferior ST-elevation myocardial infarction" in target
    assert "Aortic dissection" in target
    assert "ECG shows ST elevation" in target

    # Test metric: completely aligned explanation (mostly words from reference target/context)
    prediction = ["Inferior ST-elevation myocardial infarction occurs due to chest pain."]
    ref = [target]

    per_sample_ratio = hallucination_rate_per_sample(prediction, ref)
    assert per_sample_ratio < 0.6
    assert hallucination_rate(prediction, ref) == 0.0

    # Test metric: explanation with completely unrelated/unsupported medical terms (hallucinated)
    hallucinated_prediction = [
        "Renal papillary necrosis is characterized by sloughing of renal papillae, causing gross hematuria."
    ]
    hall_ratio = hallucination_rate_per_sample(hallucinated_prediction, ref)
    assert hall_ratio > 0.6
    assert hallucination_rate(hallucinated_prediction, ref) == 1.0
