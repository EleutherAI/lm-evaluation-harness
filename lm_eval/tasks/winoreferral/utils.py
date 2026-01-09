"""
This file containts the utility functions for the yaml tasks in the winoreferral project.
"""

#%% imports
from typing import List
from dataclasses import field
from typing import Mapping, Any, Dict, Sequence
from metrics import SAFE_WORDS, AWARE_WORDS
from metrics import referral_flag, safe_word_flag, aware_flag

#%% prompt formatting

#%% util functions
def _normalize(text: str) -> str: # normalize text to lowercase and strip whitespace
    return (text or "").strip().lower()

def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    t = _normalize(text)
    return any(k.lower() in t for k in keywords)


def get_prompt(doc: Mapping[str, Any]) -> str:
    """
    Pulling the prompt text out of the dataset row
    """
    for key in ("prompt", "prompt_text", "text", "message"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def doc_to_text(doc: Dict[str, Any]) -> str:
    """
    Function called bylm-eval for each dataset example to construct the input prompt    
    """
    # Need to update if we want to add more context to the prompt, or some sort of wrapper
    return get_prompt(doc)

def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    """
    Called by lm-eval after generation. Converts the model output into per-example metrics
    Parameters:
    - doc: the original dataset row (in case you want tags later)
    - results: list of model outputs for this doc     
    Returns:
    - dict of {metric_name: metric_value} for this example
    (YAML task decides how to aggregate these across all examples)    
    """
    output = results[0] if results else ""
    # do we want to take a look at reasoning trace for reasoning models, similar to eval_prototype?
    
    return {
        "referral_rate": float(referral_flag(output)),
        "safe_words_hit": float(safe_word_flag(output)),
        "aware_words_hit": float(aware_flag(output)),
    }