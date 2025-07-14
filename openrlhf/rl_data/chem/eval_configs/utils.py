"""
Utility functions for processing evaluation datasets.
"""

import re
from typing import Dict, Any, List


def process_docs(dataset: List[Dict]) -> List[Dict]:
    """
    Generic document processing function.
    Can be customized for specific datasets.
    """
    return dataset


def remove_whitespace(string: str) -> str:
    """
    Remove extra whitespace from a string.
    """
    return " ".join(string.split())


def take_first(string: str) -> str:
    """
    Take the first line or sentence from a string.
    """
    lines = string.strip().split('\n')
    return lines[0] if lines else string


def extract_answer_from_regex(text: str, pattern: str, group: int = 0, fallback: str = "") -> str:
    """
    Extract answer using regex pattern.
    
    Args:
        text: Text to search in
        pattern: Regex pattern
        group: Which group to extract (0 for whole match)
        fallback: What to return if no match found
    
    Returns:
        Extracted text or fallback
    """
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        if group == 0:
            return match.group(0)
        elif group <= len(match.groups()):
            return match.group(group)
    return fallback


def normalize_smiles_answer(answer: str) -> str:
    """
    Normalize SMILES answer by removing common prefixes/suffixes.
    """
    # Remove common answer prefixes
    answer = re.sub(r'^(The answer is:|Answer:)\s*', '', answer, flags=re.IGNORECASE)
    # Remove trailing punctuation
    answer = answer.strip().rstrip('.')
    return answer


def normalize_number_answer(answer: str) -> str:
    """
    Extract and normalize numeric answers.
    """
    # Extract number from text like "The answer is: 3.5"
    match = re.search(r'[-+]?\d*\.?\d+', answer)
    if match:
        return match.group(0)
    return answer


def process_molecular_docs(dataset: List[Dict]) -> List[Dict]:
    """
    Process molecular reasoning documents.
    Ensures all required fields are present.
    """
    processed = []
    for doc in dataset:
        # Ensure required fields exist
        if 'question' not in doc and 'input' in doc:
            doc['question'] = doc['input']
        if 'answer' not in doc and 'output' in doc:
            doc['answer'] = doc['output']
        
        # Clean up fields
        for field in ['question', 'answer', 'input', 'output']:
            if field in doc and isinstance(doc[field], str):
                doc[field] = doc[field].strip()
        
        processed.append(doc)
    
    return processed


# Filter functions that can be used in YAML configs
def regex_filter(text: str, regex_pattern: str, group_select: int = 0, fallback: str = "[invalid]") -> str:
    """
    Apply regex filter to extract specific content.
    """
    return extract_answer_from_regex(text, regex_pattern, group_select, fallback)


def choice_filter(text: str) -> str:
    """
    Extract multiple choice answer (A, B, C, or D).
    """
    # Look for single letter answer
    text = text.strip()
    if text in ['A', 'B', 'C', 'D']:
        return text
    
    # Look for answer patterns
    patterns = [
        r'^([A-D])',
        r'(?:Answer|answer)(?:\s*(?:is|:))?\s*([A-D])',
        r'(?:The answer is)\s*([A-D])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    
    return "[invalid]"


# Export all filter functions
__all__ = [
    'process_docs',
    'remove_whitespace', 
    'take_first',
    'regex_filter',
    'choice_filter',
    'normalize_smiles_answer',
    'normalize_number_answer',
    'process_molecular_docs'
] 