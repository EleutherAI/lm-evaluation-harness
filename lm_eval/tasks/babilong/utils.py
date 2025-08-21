"""Utility functions for Babilong evaluation."""

import re
import string
from typing import List, Dict, Any


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def extract_answer(text: str) -> str:
    """Extract answer from model output."""
    # Try to find answer after "Answer:" or "A:" patterns
    patterns = [
        r'answer[:\s]+([^\n]+)',
        r'a[:\s]+([^\n]+)',
        r'^([^\n]+)$'  # If no pattern, take first line
    ]
    
    text = text.strip()
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Fallback: return the whole text
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """Check if prediction exactly matches ground truth after normalization."""
    pred_normalized = normalize_answer(extract_answer(prediction))
    truth_normalized = normalize_answer(ground_truth)
    
    return 1.0 if pred_normalized == truth_normalized else 0.0


def contains_answer(prediction: str, ground_truth: str) -> float:
    """Check if prediction contains the ground truth answer."""
    pred_normalized = normalize_answer(extract_answer(prediction))
    truth_normalized = normalize_answer(ground_truth)
    
    return 1.0 if truth_normalized in pred_normalized else 0.0


def numeric_match(prediction: str, ground_truth: str, tolerance: float = 0.001) -> float:
    """Check if prediction matches a numeric answer within tolerance."""
    try:
        # Extract numbers from both strings
        pred_nums = re.findall(r'-?\d+\.?\d*', extract_answer(prediction))
        truth_nums = re.findall(r'-?\d+\.?\d*', ground_truth)
        
        if not pred_nums or not truth_nums:
            # Fall back to exact match if no numbers found
            return exact_match(prediction, ground_truth)
        
        pred_num = float(pred_nums[0])
        truth_num = float(truth_nums[0])
        
        return 1.0 if abs(pred_num - truth_num) < tolerance else 0.0
    except:
        # Fall back to exact match if parsing fails
        return exact_match(prediction, ground_truth)


def yes_no_match(prediction: str, ground_truth: str) -> float:
    """Check if yes/no answer matches."""
    pred = extract_answer(prediction).lower()
    truth = ground_truth.lower()
    
    # Look for yes/no patterns
    yes_patterns = ['yes', 'true', 'correct', 'right', 'affirmative']
    no_patterns = ['no', 'false', 'incorrect', 'wrong', 'negative']
    
    pred_is_yes = any(pattern in pred for pattern in yes_patterns)
    pred_is_no = any(pattern in pred for pattern in no_patterns)
    truth_is_yes = any(pattern in truth for pattern in yes_patterns)
    truth_is_no = any(pattern in truth for pattern in no_patterns)
    
    if (pred_is_yes and truth_is_yes) or (pred_is_no and truth_is_no):
        return 1.0
    
    # Fall back to exact match
    return exact_match(prediction, ground_truth)


def list_match(prediction: str, ground_truth: str) -> float:
    """Check if prediction matches a list of items."""
    pred = extract_answer(prediction).lower()
    truth = ground_truth.lower()
    
    # Extract comma or space separated items
    pred_items = set(re.split(r'[,\s]+', pred))
    truth_items = set(re.split(r'[,\s]+', truth))
    
    # Remove empty strings
    pred_items = {item for item in pred_items if item}
    truth_items = {item for item in truth_items if item}
    
    if not pred_items or not truth_items:
        return exact_match(prediction, ground_truth)
    
    # Check if sets match
    return 1.0 if pred_items == truth_items else 0.0


# Task-specific metric mappings
TASK_METRICS = {
    'qa1': exact_match,
    'qa2': exact_match,
    'qa3': exact_match,
    'qa4': exact_match,
    'qa5': exact_match,
    'qa6': yes_no_match,
    'qa7': numeric_match,
    'qa8': list_match,
    'qa9': exact_match,
    'qa10': exact_match,
    'qa11': exact_match,
    'qa12': exact_match,
    'qa13': exact_match,
    'qa14': exact_match,
    'qa15': exact_match,
    'qa16': exact_match,
    'qa17': exact_match,
    'qa18': exact_match,
    'qa19': list_match,
    'qa20': exact_match,
}


def get_metric_for_task(task_name: str):
    """Get appropriate metric function for a Babilong task."""
    # Extract task number from name (e.g., qa1, qa2, etc.)
    match = re.search(r'qa(\d+)', task_name.lower())
    if match:
        task_key = f'qa{match.group(1)}'
        return TASK_METRICS.get(task_key, exact_match)
    return exact_match


def process_results_gen(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    """Process generation results for Babilong tasks."""
    prediction = results[0] if results else ""
    
    # Get ground truth answer
    answer = doc.get("answer", "")
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    
    # Get appropriate metric for the task
    task_name = doc.get("task_name", "")
    metric_fn = get_metric_for_task(task_name)
    
    # Compute score
    score = metric_fn(prediction, str(answer))
    
    return {"acc": score}


def create_prompt(doc: Dict[str, Any]) -> str:
    """Create prompt for Babilong tasks."""
    context = doc.get("context", "")
    question = doc.get("question", "")
    
    prompt = f"""Read the following text carefully and answer the question based only on the information provided.

Context:
{context}

Question: {question}
Answer:"""
    
    return prompt