import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process the AIBE dataset for evaluation."""
    
    def _process_doc(doc):
        # Make sure the correct option is just the letter (A, B, C, or D)
        correct_option = doc['correct_option']
        # Ensure it's just the letter
        if correct_option and len(correct_option) > 1:
            correct_option = correct_option[0]
            
        return {
            'question': doc['question'],
            'option_a': doc['option_a'],
            'option_b': doc['option_b'],
            'option_c': doc['option_c'],
            'option_d': doc['option_d'],
            'correct_option': correct_option,
        }
    
    return dataset.map(_process_doc)