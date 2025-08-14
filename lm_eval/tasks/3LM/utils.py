import re
PROMPT_TEMPLATE = (
    "You are an expert in {domain}. Analyze the given multiple-choice question and "
    "provide the correct answer using this approach:\n"
    "1. Carefully read the question and options\n"
    "2. Identify core {domain} concepts and required knowledge\n"
    "3. Analyze each option for relevance, accuracy, and consistency\n"
    "4. Consider {domain}-specific context and factors\n"
    "5. Use elimination and comparative analysis\n"
    "6. Select the most accurate answer\n"
    "Maintain objectivity, consider {domain}-specific sensitivities, and base your decision "
    "on verifiable facts and sound logical reasoning within {domain}.\n"
    "Question: {question}\n"
    "{choices_text}\n"
    "Correct option number is:"
)

def doc_to_text(doc):
    choices = doc["choices"]
    choices_text = "\n".join(choice.replace("(أ)", "أ.")  
                                   .replace("(ب)", "ب.")  
                                   .replace("(ج)", "ج.")  
                                   .replace("(د)", "د.") 
                                   for choice in choices)

    return PROMPT_TEMPLATE.format(
        domain=doc["domain"],
        question=doc["question_text"],
        choices_text=choices_text
    )


def extract_choices(choice_str):
    """
    Extract all text between single quotes as separate options
    and clean up extra ') ' after the option letter.
    """
    options = re.findall(r"'(.*?)'", choice_str)
    cleaned_options = [opt.replace(") )", ")").strip() for opt in options]
    return cleaned_options

def make_filter(domain):
    def filter_fn(doc):
        doc['domain'] = domain
        doc['choices'] = extract_choices(doc['choices'])
        doc['correct_choice'] = doc['self_answer']
        doc['question_text'] = doc['question']

        return doc_to_text(doc)
    return filter_fn

doc_to_text_biology = make_filter("Biology")
doc_to_text_physics= make_filter("Physics")
doc_to_text_math = make_filter("Math")
doc_to_text_chemistry = make_filter("Chemistry")
doc_to_text_general_science = make_filter("General_Science")

def doc_to_choice(doc):
    """Return non-empty choices in order."""
    choices = ["أ", "ب", "ج", "د"]
    return choices

def doc_to_target(doc):
    """Return the correct option number as a string if the option is non-empty."""
    doc['correct_choice'] = doc['self_answer']
    correct = doc.get("correct_choice")
    return str(correct)
