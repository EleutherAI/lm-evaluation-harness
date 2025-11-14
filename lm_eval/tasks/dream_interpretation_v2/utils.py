def doc_to_choice(doc):
    """Extract the choice labels from the options."""
    options = doc["options"]

    if isinstance(options, dict):
        # Format: {"A": "text", "B": "text", ...}
        return list(options.keys())
    elif isinstance(options, list):
        # Format: ["A) text", "B) text", ...] or ["A. text", "B. text", ...]
        return [option.split(")")[0].split(".")[0].strip() for option in options]
    else:
        return ["A", "B", "C", "D", "E"]  # fallback


def doc_to_target(doc):
    """Get the target index for the correct answer."""
    choices = doc_to_choice(doc)
    correct_answer = doc["answer"]  # Changed from "correct_answer" to "answer"

    # Handle both Arabic and English choice labels
    if correct_answer in choices:
        return choices.index(correct_answer)
    else:
        # Fallback: try to find by letter mapping
        choice_mapping = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "أ": 0,
            "ب": 1,
            "ج": 2,
            "د": 3,
            "هـ": 4,
        }
        return choice_mapping.get(correct_answer, 0)


# def doc_to_text(doc):
#     """Format the question and options for display."""
#     question = doc["question"]
#     options = doc["options"]

#     formatted_text = f"### Question\n{question}\n\n### Options\n"

#     if isinstance(options, dict):
#         # Format: {"A": "text", "B": "text", ...}
#         for key, value in options.items():
#             formatted_text += f"{key}: {value}\n"
#     elif isinstance(options, list):
#         # Format: ["A) text", "B) text", ...]
#         for option in options:
#             formatted_text += f"{option}\n"

#     formatted_text += "\n============="

#     return formatted_text
