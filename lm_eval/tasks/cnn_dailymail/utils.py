"""
Utility functions for CNN/DailyMail summarization task evaluation.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List


try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not installed. Install with: pip install rouge-score")

try:
    from bert_score import score as bert_score

    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert_score not installed. Install with: pip install bert-score")


def normalize_text(text: str) -> str:
    """
    Normalize text for evaluation by removing extra whitespace and lowercasing.

    Args:
        text: Input text string

    Returns:
        Normalized text string
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def calculate_rouge_scores(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """
    Calculate ROUGE scores for a list of predictions and references.

    Args:
        predictions: List of generated summaries
        references: List of reference summaries

    Returns:
        Dictionary with rouge1, rouge2, and rougeL scores
    """
    if not ROUGE_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        pred = normalize_text(pred)
        ref = normalize_text(ref)

        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        "rougeL": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
    }


def calculate_bertscore(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """
    Calculate BERTScore for a list of predictions and references.

    Args:
        predictions: List of generated summaries
        references: List of reference summaries

    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    if not BERTSCORE_AVAILABLE:
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
        }

    # Normalize texts
    predictions = [normalize_text(p) for p in predictions]
    references = [normalize_text(r) for r in references]

    # Calculate BERTScore
    # Using distilbert-base-uncased for faster computation
    # You can change to 'roberta-large' for better quality
    P, R, F1 = bert_score(
        predictions,
        references,
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
    )

    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    """
    Process results for a single document.

    This function is called by lm-eval-harness for each document after generation.

    Args:
        doc: The document dictionary containing 'highlights' (reference summary)
        results: List containing the generated text(s)

    Returns:
        Dictionary with metric scores for this document
    """
    # Get the generated summary
    if not results or len(results) == 0:
        generated_summary = ""
    else:
        generated_summary = results[0] if isinstance(results, list) else results

    # Get the reference summary
    reference_summary = doc.get("highlights", "")

    # Normalize both texts
    generated_summary = normalize_text(generated_summary)
    reference_summary = normalize_text(reference_summary)

    # Calculate ROUGE scores
    rouge_results = {}
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        scores = scorer.score(reference_summary, generated_summary)
        rouge_results = {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    # Calculate BERTScore
    bertscore_results = {}
    if BERTSCORE_AVAILABLE:
        P, R, F1 = bert_score(
            [generated_summary],
            [reference_summary],
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
        )
        bertscore_results = {
            "bertscore_precision": P[0].item(),
            "bertscore_recall": R[0].item(),
            "bertscore_f1": F1[0].item(),
        }

    # Calculate summary length
    summary_length = calculate_summary_length(generated_summary)

    # Combine all results
    return {**rouge_results, **bertscore_results, "summary_length": summary_length}


def postprocess_generation(generation: str) -> str:
    """
    Post-process the generated text to clean it up.

    Args:
        generation: Raw generated text

    Returns:
        Cleaned generated text
    """
    # Remove leading/trailing whitespace
    generation = generation.strip()

    # Remove any repeated newlines
    generation = re.sub(r"\n+", " ", generation)

    # Remove extra spaces
    generation = re.sub(r"\s+", " ", generation)

    return generation


def filter_long_articles(doc: Dict[str, Any]) -> bool:
    """
    Filter out articles that are too long.

    Args:
        doc: Document dictionary

    Returns:
        True if document should be kept, False if it should be filtered out
    """
    article = doc.get("article", "")
    # Filter out articles longer than 2000 words
    word_count = len(article.split())
    return word_count <= 2000


def doc_to_choice(doc: Dict[str, Any]) -> List[str]:
    """
    For multiple-choice format (not used in summarization, but kept for compatibility).

    Args:
        doc: Document dictionary

    Returns:
        List of choices (empty for generation tasks)
    """
    return []


def process_docs(dataset):
    """
    Pre-process the entire dataset before evaluation.

    Args:
        dataset: HuggingFace dataset

    Returns:
        Processed dataset
    """

    def _process_doc(doc):
        """Process a single document."""
        # Clean the article text
        article = doc.get("article", "")
        article = normalize_text(article)

        # Clean the highlights
        highlights = doc.get("highlights", "")
        highlights = normalize_text(highlights)

        return {
            **doc,
            "article": article,
            "highlights": highlights,
            "article_length": len(article.split()),
            "summary_length": len(highlights.split()),
        }

    return dataset.map(_process_doc)


def calculate_summary_length(generated: str) -> int:
    """
    Calculate the length of generated summary in words.

    Args:
        generated: Generated summary text

    Returns:
        Number of words in summary
    """
    return len(normalize_text(generated).split())
