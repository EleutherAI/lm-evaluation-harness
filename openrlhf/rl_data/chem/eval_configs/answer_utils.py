"""
Answer extraction and matching utilities for molecular reasoning tasks.
This module provides functions for extracting answers from various formats,
validating SMILES strings, calculating molecular similarity, and normalizing answers.
"""

import re
import string
from typing import List, Optional, Tuple, Union

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Regular expression patterns for answer extraction
ANSWER_TAG_PATTERN = r'<answer>(.*?)</answer>'
ANSWER_COLON_PATTERN = r'(?:answer|result|solution)(?:\s+is|:)\s*(.*?)(?:\.|$|\n)'
BOXED_PATTERN = r'\\boxed{(.*?)}'
FINAL_ANSWER_PATTERN = r'final answer(?:\s+is|:)\s*(.*?)(?:\.|$|\n)'

def extract_answer_from_text(text: str) -> str:
    """
    Extract answer from text using various patterns.
    
    Args:
        text: The text to extract the answer from
        
    Returns:
        The extracted answer or the original text if no pattern matches
    """
    if not text:
        return ""
    
    # Try to extract from <answer> tags first
    match = re.search(ANSWER_TAG_PATTERN, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Try to extract from \boxed{} notation
    match = re.search(BOXED_PATTERN, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try to extract from "answer is: X" format
    match = re.search(ANSWER_COLON_PATTERN, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Try to extract from "final answer: X" format
    match = re.search(FINAL_ANSWER_PATTERN, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # If no pattern matches, return the original text
    return text.strip()

def extract_smiles_from_text(text: str) -> List[str]:
    """
    Extract and validate SMILES strings from text.
    
    Args:
        text: The text to extract SMILES from
        
    Returns:
        List of valid, canonicalized SMILES strings
    """
    if not RDKIT_AVAILABLE:
        return []
    
    # First extract the answer
    answer = extract_answer_from_text(text)
    
    # Look for SMILES patterns
    smiles_pattern = r'([A-Za-z0-9\(\)\[\]\.\=\#\@\-\\\/\+]+)'
    matches = re.findall(smiles_pattern, answer)
    
    valid_smiles = []
    for match in matches:
        candidate = match.strip()
        if is_valid_smiles(candidate):
            valid_smiles.append(canonicalize_smiles(candidate))
    
    return valid_smiles

def is_valid_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is valid.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not RDKIT_AVAILABLE:
        return False
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def canonicalize_smiles(smiles: str) -> str:
    """
    Convert SMILES to canonical form.
    
    Args:
        smiles: SMILES string to canonicalize
        
    Returns:
        Canonicalized SMILES string or empty string if invalid
    """
    if not RDKIT_AVAILABLE:
        return smiles
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except:
        return ""

def calculate_molecular_similarity(smiles1: str, smiles2: str) -> float:
    """
    Calculate Tanimoto similarity between two molecules.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        
    Returns:
        Tanimoto similarity (0-1) or -1 if invalid SMILES
    """
    if not RDKIT_AVAILABLE:
        return -1
    
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return -1
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return -1

def normalize_answer(answer: str) -> str:
    """
    Normalize answer by removing punctuation and converting to lowercase.
    
    Args:
        answer: The answer to normalize
        
    Returns:
        Normalized answer
    """
    if not answer:
        return ""
    
    # Remove punctuation and convert to lowercase
    translator = str.maketrans("", "", string.punctuation)
    normalized = answer.translate(translator).lower()
    
    # Remove extra whitespace
    normalized = " ".join(normalized.split())
    
    return normalized

def extract_multiple_choice_answer(text: str) -> str:
    """
    Extract a single letter answer (A, B, C, D) from text.
    
    Args:
        text: The text to extract the answer from
        
    Returns:
        The extracted letter or empty string if not found
    """
    # First try to extract from standard patterns
    answer = extract_answer_from_text(text)
    
    # Look for a single letter answer
    match = re.search(r'\b([A-D])\b', answer)
    if match:
        return match.group(1).upper()
    
    # If not found in the extracted answer, search the whole text
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1).upper()
    
    return ""

def check_answer_match(prediction: str, reference: str, 
                      ignore_case: bool = True, 
                      ignore_punctuation: bool = False) -> bool:
    """
    Check if prediction matches reference answer.
    
    Args:
        prediction: The predicted answer
        reference: The reference answer
        ignore_case: Whether to ignore case differences
        ignore_punctuation: Whether to ignore punctuation differences
        
    Returns:
        True if answers match, False otherwise
    """
    if ignore_case:
        prediction = prediction.lower()
        reference = reference.lower()
    
    if ignore_punctuation:
        prediction = normalize_answer(prediction)
        reference = normalize_answer(reference)
    
    return prediction.strip() == reference.strip()

def extract_numerical_answer(text: str) -> Optional[float]:
    """
    Extract a numerical answer from text.
    
    Args:
        text: The text to extract the numerical answer from
        
    Returns:
        The extracted number or None if not found
    """
    # First extract the answer
    answer = extract_answer_from_text(text)
    
    # Look for a number
    match = re.search(r'[-+]?\d*\.?\d+', answer)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    
    return None 