import re


def wikitext_detokenizer(doc):
    string = doc["label"]
    string = string.replace('[.,]', '').lower()
    string = string.split("\\n\\n")
    string = string.split("<pad>")[-1].split("</s>")[0].strip()
    string = extract_answer(string)
    string = verbalizer(string.strip())
    return string


def extract_answer(string):
    pattern = r'(\*\*answer:\*\*|\*answer is:\*|\*\*|\*\*|\*answer is exact\*|label:|the premise and hypothesis ' \
              r'are|the premise and the hypothesis is|the premise and the hypothesis is a|described as|therefore they ' \
              r'are|therefore|are considered|is an exact|it is|is a|is)\s*(neutral|entailment|contradiction)'
    match = re.search(pattern, string, re.IGNORECASE)
    return match.group(2) if match else string


def verbalizer(string):
    verbalizer_dict = {
        "entailment": ['encouragement', 'entitlement', 'entails', 'entailed', 'entailment'],
        "contradiction": ['contradictory', 'contradicts', 'contradiction'],
        "neutral": ['neutral']}
    for key, values in verbalizer_dict.items():
        for value in values:
            if value in string:
                return key
    return string

