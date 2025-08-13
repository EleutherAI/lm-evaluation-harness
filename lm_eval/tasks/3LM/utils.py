import re

def doc_to_text(doc):
    question = doc["question_text"]
    choices = doc["choices"]
    domain = doc["domain"]
    choices_text = "\n".join(choice.replace("(أ)", "أ.")  
                                   .replace("(ب)", "ب.")  
                                   .replace("(ج)", "ج.")  
                                   .replace("(د)", "د.") 
                                   for choice in choices)

    prompt = (
        f"You are an expert in {domain}. Analyze the given multiple-choice question and "
        f"provide the correct answer using this approach:\n"
        f"1. Carefully read the question and options\n"
        f"2. Identify core {domain} concepts and required knowledge\n"
        f"3. Analyze each option for relevance, accuracy, and consistency\n"
        f"4. Consider {domain}-specific context and factors\n"
        f"5. Use elimination and comparative analysis\n"
        f"6. Select the most accurate answer\n"
        f"Maintain objectivity, consider {domain}-specific sensitivities, and base your decision on verifiable facts "
        f"and sound logical reasoning within {domain}.\n"
        f"Question: {question}\n{choices_text}\nCorrect option number is:"
    )
    return prompt

def doc_to_choice(doc):
    """Return non-empty choices in order."""
    choices = ["أ", "ب", "ج", "د"]
    return choices

def doc_to_target(doc):
    """Return the correct option number as a string if the option is non-empty."""
    correct = doc.get("correct_choice")
    return str(correct)
