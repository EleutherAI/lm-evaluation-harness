import re

mcq_user_prompt_template = """
## Question
{question}

## Options
{choices}

=========
Answer the above question by only returning the option (A, B, C, D or E) without any further explanation.
"""


def doc_to_text(doc):
    """Format the question and options for generation-based evaluation."""
    question = doc["question"]
    options = doc["options"]

    # Format choices string
    choices = ""
    if isinstance(options, dict):
        # Format: {"A": "text", "B": "text", ...}
        for key, value in options.items():
            choices += f"{key}: {value}\n"
    elif isinstance(options, list):
        # Format: ["A) text", "B) text", ...]
        for option in options:
            choices += f"{option}\n"

    # Use the template
    return mcq_user_prompt_template.format(question=question, choices=choices.strip())


def doc_to_target(doc):
    """Get the target answer for the correct choice."""
    return doc["answer"]


def extract_answer(generation):
    """
    Extract the answer choice (A, B, C, D, E) from the model's generation.
    Handles both English and Arabic choice labels.
    """
    generation = generation.strip()

    # First, try to find exact matches for choice letters at the beginning
    english_pattern = r"^([A-E])[^A-Za-z]"
    arabic_pattern = r"^([أ-هـ])[^أ-ي]"

    # Try English letters first
    match = re.search(english_pattern, generation)
    if match:
        return match.group(1)

    # Try Arabic letters
    match = re.search(arabic_pattern, generation)
    if match:
        arabic_to_english = {"أ": "A", "ب": "B", "ج": "C", "د": "D", "هـ": "E"}
        return arabic_to_english.get(match.group(1), match.group(1))

    # Look for letters anywhere in the text (more flexible)
    # English letters
    for letter in ["A", "B", "C", "D", "E"]:
        if letter in generation:
            return letter

    # Arabic letters
    arabic_letters = {"أ": "A", "ب": "B", "ج": "C", "د": "D", "هـ": "E"}
    for arabic, english in arabic_letters.items():
        if arabic in generation:
            return english

    # If no clear match found, return the first character if it's a valid choice
    first_char = generation[0] if generation else ""
    if first_char in ["A", "B", "C", "D", "E"]:
        return first_char
    if first_char in arabic_letters:
        return arabic_letters[first_char]

    # Default fallback
    return "Z"  # Indicate no valid answer found


def post_process_response(text):
    # remove unnecessary content
    text = text.split("</think>")[-1].strip()
    unncessary_content = ["Answer:", "<think>\n\n</think>", "<think>", "</think>"]
    for content in unncessary_content:
        text = text.replace(content, "")
    return text.strip()


def process_results(doc, results):
    """
    Process generation results to calculate accuracy.
    """

    # print(f"results: {results}")
    # print(f"doc: {doc}")
    generation = results[0] if results else ""

    # print(f"generation: {generation}")
    generation = post_process_response(generation)
    # print(f"post-processed generation: {generation}")
    predicted_answer = extract_answer(generation)
    target_answer = doc_to_target(doc)

    # print(f"Predicted Answer: {predicted_answer}, Target Answer: {target_answer}")
    # Calculate accuracy
    acc = 1.0 if predicted_answer == target_answer else 0.0

    return {"acc": acc}
