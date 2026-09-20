import random

def doc_to_text(doc):
    def format_example(doc, keys):
        prompt = f"This is a {doc['Subject']} question for {doc['Level']}. Please choose the correct answer!\n\n"
        prompt += "Question: " + doc["Question"] + "\n"
        prompt += "".join(
            [f"Option {key}: {choice}\n" for key, choice in zip(keys, options) if choice]
        )
        prompt += "Answer:"
        return prompt
    
    options = [doc.get(f"Option {key}", "").strip() if doc.get(f"Option {key}") is not None else "" for key in ["A", "B", "C", "D", "E"]]

        # Determine which options are non-empty
    num_choices = len([opt for opt in options if opt])  # Count non-empty options
    keys = ["A", "B", "C", "D", "E"][:num_choices]

    return format_example(doc, keys)


def doc_to_choice(doc):
    """
    Extracts valid multiple-choice options from the document.
    """
    options = [doc.get(f"Option {key}", "").strip() if doc.get(f"Option {key}") is not None else "" for key in ["A", "B", "C", "D", "E"]]
    num_choices = len([opt for opt in options if opt]) 

    return options[:num_choices]


def doc_to_target(doc):
    """
    Extracts the correct answer's index from the given document.
    """
    options = [doc.get(f"Option {key}", "").strip() if doc.get(f"Option {key}") is not None else "" for key in ["A", "B", "C", "D", "E"]]

    # Determine which options are non-empty
    num_choices = len([opt for opt in options if opt])  # Count non-empty options
    keys = ["A", "B", "C", "D", "E"][:num_choices]

    return keys.index(doc["Answer Key"].strip())


def doc_to_decontamination_query(doc):
    """
    Provides a query for decontamination (checking if similar text exists in pretraining data).
    """
    return doc_to_text(doc)