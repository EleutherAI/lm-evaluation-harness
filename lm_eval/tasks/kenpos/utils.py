"""
KenPOS Dataset Utilities
Helper functions for processing KenPOS POS tagging data
"""


def doc_to_text(doc):
    """Convert document to input text for the model"""
    word = doc.get("token", "")
    
    prompt = f"""Tag the following word with its part of speech.

Word: {word}
POS Tag:"""
    
    return prompt


def doc_to_target(doc):
    """Extract the target POS tag from the document"""
    pos_tag = doc.get("pos_tag", "")
    return f" {pos_tag}"


def process_results(doc, results):
    """Process the model's generated results and calculate metrics"""
    pred = results[0].strip() if results else ""
    pred_tag = pred.split()[0] if pred.split() else pred
    target = doc.get("pos_tag", "")
    
    # CRITICAL: Return only numeric values for metrics
    exact_match = 1.0 if pred_tag.upper() == target.upper() else 0.0
    
    # Only return the metric, not the pred/target strings
    return {
        "exact_match": exact_match
    }


def doc_to_decontamination_query(doc):
    """Generate decontamination query"""
    word = doc.get("token", "")
    return word