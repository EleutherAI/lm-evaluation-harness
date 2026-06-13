# -*- coding: utf-8 -*-
"""Clinical hallucination detection metrics for lm-evaluation-harness.

Measures the hallucination rate in LLM responses to clinical QA questions by
extracting medical terms from model predictions and checking what fraction
are unsupported by the reference answer.

A response is flagged as hallucinated when >60% of its extracted medical terms
are NOT present in the reference text.

Based on methodology from clinical-llm-eval hallucination detection.
"""

import re
from typing import List, Set


# ---------------------------------------------------------------------------
# Medical term extraction
# ---------------------------------------------------------------------------

# Regex patterns for extracting clinical/medical terms
_MEDICAL_PATTERNS = [
    # Medication dosages and units (e.g. "500 mg", "0.5 mcg", "100 mmHg")
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|μg|µg|ml|ML|L|mmol|mmHg|bpm|"
        r"mEq|IU|units?|mg/kg|mg/dL|mmol/L|mL/min|g/dL)\b",
        re.IGNORECASE,
    ),
    # Standalone medical units
    re.compile(
        r"\b(?:mg|mcg|μg|µg|ml|L|mmol|mmHg|bpm|mEq|IU)\b",
        re.IGNORECASE,
    ),
    # Drug name patterns – common pharmacological suffixes
    re.compile(
        r"\b[A-Z][a-z]+(?:ine|ol|an|ide|ate|ase|ium|ine|epam|statin|cillin|mycin|cycline|nib|mab)\b"
    ),
    # Disease staging / classification
    re.compile(
        r"\b(?:type\s*[0-9IV]+|stage\s*[IV0-9]+|grade\s*[0-9IV]+|"
        r"class\s*[IV0-9]+|phase\s*[IV0-9]+)\b",
        re.IGNORECASE,
    ),
    # Proper nouns (capitalized multi-word terms – often disease/syndrome names)
    re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"),
    # Common medical abbreviations (e.g. COPD, MI, CHF, DKA)
    re.compile(r"\b[A-Z]{2,6}\b"),
    # Lab value patterns (e.g. "WBC 11.2", "HbA1c 7.5%")
    re.compile(
        r"\b(?:WBC|RBC|Hgb|Hct|PLT|BUN|Cr|AST|ALT|ALP|GGT|LDH|CK|"
        r"Troponin|BNP|D-dimer|INR|PT|PTT|aPTT|HbA1c|FBS|CBC|CMP|"
        r"BMP|TSH|T3|T4|PSA|CEA|CA-?\d+)\b",
        re.IGNORECASE,
    ),
]

# Static set of medical keywords that are always considered "medical terms"
MEDICAL_KEYWORDS: Set[str] = {
    "diagnosis", "treatment", "prognosis", "medication", "surgery",
    "therapy", "infection", "inflammation", "chronic", "acute",
    "benign", "malignant", "biopsy", "metastasis", "carcinoma",
    "sarcoma", "lymphoma", "leukemia", "pathogen", "antibody",
    "antigen", "cytokine", "receptor", "enzyme", "inhibitor",
    "agonist", "antagonist", "contraindication", "indication",
    "adverse", "efficacy", "potency", "bioavailability",
    "pharmacokinetics", "pharmacodynamics", "dosage", "regimen",
    "prophylaxis", "comorbidity", "etiology", "pathogenesis",
    "histology", "cytology", "necrosis", "apoptosis", "thrombosis",
    "embolism", "ischemia", "infarction", "stenosis", "aneurysm",
    "hypertension", "hypotension", "tachycardia", "bradycardia",
    "arrhythmia", "edema", "fibrosis", "cirrhosis", "hepatitis",
    "nephritis", "pancreatitis", "meningitis", "encephalitis",
    "pneumonia", "bronchitis", "sepsis", "abscess", "ulcer",
    "hemorrhage", "anemia", "thrombocytopenia", "neutropenia",
}


def _extract_medical_terms(text: str) -> Set[str]:
    """Extract a set of medical terms from *text*.

    Combines regex-pattern matches with keyword detection to build
    a comprehensive term set.  All tokens are lower-cased for
    case-insensitive comparison.
    """
    terms: Set[str] = set()

    # Regex-based extraction
    for pattern in _MEDICAL_PATTERNS:
        for match in pattern.finditer(text):
            terms.add(match.group().lower().strip())

    # Keyword extraction – check every whitespace-delimited token
    for token in text.lower().split():
        # Strip trailing punctuation for matching
        cleaned = re.sub(r"[^\w\s]", "", token)
        if cleaned in MEDICAL_KEYWORDS:
            terms.add(cleaned)

    return terms


# Threshold: a response is flagged as hallucinated when more than this
# fraction of its extracted terms are absent from the reference.
HALLUCINATION_THRESHOLD = 0.6


def _is_hallucinated(reference: str, prediction: str) -> float:
    """Return 1.0 if *prediction* is hallucinated, else 0.0.

    A prediction is considered hallucinated when >60 % of its extracted
    medical terms do not appear in the reference text.
    """
    pred_terms = _extract_medical_terms(prediction)
    if not pred_terms:
        # No medical terms extracted – cannot assess hallucination
        return 0.0

    ref_terms = _extract_medical_terms(reference)
    # Also do a simple substring check against the lower-cased reference
    # to catch terms that were split or embedded differently.
    ref_lower = reference.lower()

    unsupported = 0
    for term in pred_terms:
        # A term is "supported" if it appears either in the ref term set
        # OR as a substring of the reference text.
        if term not in ref_terms and term not in ref_lower:
            unsupported += 1

    unsupported_ratio = unsupported / len(pred_terms)
    return 1.0 if unsupported_ratio > HALLUCINATION_THRESHOLD else 0.0


def _unsupported_term_ratio(reference: str, prediction: str) -> float:
    """Return the fraction of prediction terms not found in reference."""
    pred_terms = _extract_medical_terms(prediction)
    if not pred_terms:
        return 0.0

    ref_terms = _extract_medical_terms(reference)
    ref_lower = reference.lower()

    unsupported = sum(
        1 for term in pred_terms
        if term not in ref_terms and term not in ref_lower
    )
    return unsupported / len(pred_terms)


# ---------------------------------------------------------------------------
# Metric functions (called by lm-evaluation-harness via !function in YAML)
# ---------------------------------------------------------------------------

def hallucination_rate(predictions: List[str], references: List[str]) -> float:
    """Compute per-sample hallucination flag (0.0 or 1.0).

    Called once per (prediction, reference) pair by the harness.
    The YAML ``aggregation: mean`` then averages these to produce
    the overall hallucination rate across the dataset.

    Parameters
    ----------
    predictions : list[str]
        Model-generated text (first element used).
    references : list[str]
        Reference / ground-truth text (first element used).

    Returns
    -------
    float
        1.0 if the prediction is hallucinated, 0.0 otherwise.
    """
    return _is_hallucinated(references[0], predictions[0])


def hallucination_rate_per_sample(
    predictions: List[str], references: List[str]
) -> float:
    """Return the raw unsupported-term ratio for a single sample.

    This gives a continuous score [0.0, 1.0] representing the fraction
    of medical terms in the prediction that are absent from the reference,
    enabling finer-grained per-sample analysis.

    Parameters
    ----------
    predictions : list[str]
        Model-generated text (first element used).
    references : list[str]
        Reference / ground-truth text (first element used).

    Returns
    -------
    float
        Ratio of unsupported medical terms (0.0 = fully grounded,
        1.0 = all terms unsupported).
    """
    return _unsupported_term_ratio(references[0], predictions[0])
