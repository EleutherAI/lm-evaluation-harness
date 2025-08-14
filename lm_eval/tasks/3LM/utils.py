import re
import ast

def doc_to_text(query):
    return query

def doc_to_choice(doc):
    return ["أ", "ب", "ج", "د"]

def doc_to_target(doc):
    try:
        doc['correct_choice'] = doc['self_answer']
    except:
        pass
    correct = doc.get("correct_choice").strip()
    return str(correct)



#Native
def process_docs(domain):
    def filter_fn(dataset):
        return dataset.filter(lambda doc: doc.get("domain") == domain)
    return filter_fn

process_docs_biology = process_docs("Biology")
process_docs_math = process_docs("Math")
process_docs_physics = process_docs("Physics")
process_docs_chemistry = process_docs("Chemistry")
process_docs_geography = process_docs("Geography")

def native_parse_choices(doc):
    raw_choices_input = doc["choices"]
    raw_choices = ast.literal_eval(raw_choices_input) if isinstance(raw_choices_input, str) else raw_choices_input
    labels, texts = [], []
    for i, choice in enumerate(raw_choices):
        match = re.match(r"^\((.)\)\s*(.*)", choice)
        if match:
            labels.append(match.group(1).strip())
            texts.append(match.group(2).strip())
        else:
            raise ValueError(f"Malformed choice: {choice}")
    labels = [re.sub(r"^\)?\s*", "", l).strip(" []'\"\n") for l in labels]
    return labels, texts

def doc_to_text_native(doc):
    labels, texts = native_parse_choices(doc)
    instruction = "السؤال التالي هو سؤال متعدد الخيارات. اختر الإجابة الصحيحة:\n\n"
    question_text = doc["question_text"].strip()

    query = f"{instruction}{question_text}\n"
    query += "".join([f"{label}. {text}\n" for label, text in zip(labels, texts)])
    query += "الإجابة:"

    return doc_to_text(query)

#Synthetic
# def extract_choices(choice_str):
#     """
#     Extract all text between single quotes as separate options
#     and clean up extra ') ' after the option letter.
#     """
#     options = re.findall(r"'(.*?)'", choice_str)
#     cleaned_options = [opt.replace(") )", ")").strip() for opt in options]
#     return cleaned_options

def synthetic_parse_choices(raw, labels):
    if all(lbl in raw for lbl in labels):
        positions_and_labels = sorted((raw.find(lbl), lbl) for lbl in labels if raw.find(lbl) != -1)
        label_to_text = {
            lbl: raw[pos + len(lbl): positions_and_labels[i + 1][0] - 1 if i < 3 else len(raw)].strip()
            for i, (pos, lbl) in enumerate(positions_and_labels)
        }
        return [label_to_text[lbl] for lbl in labels]
    elif "," in raw:
        parts, buffer, depth = [], "", 0
        for ch in raw:
            if ch == "(": depth += 1
            elif ch == ")": depth = max(depth - 1, 0)
            if ch == "," and depth == 0:
                parts.append(buffer.strip()); buffer = ""
            else:
                buffer += ch
        parts.append(buffer.strip())

        if len(parts) != 4:
            raise ValueError(f"Expected 4 top-level commas, got {len(parts)}: {parts!r}")

        return [part[2:].strip() for part in parts]
    else:
        raise ValueError(f"Cannot determine how to split choices: {raw!r}")


def make_filter_synthetic(domain):
    def filter_fn(doc):
        labels = ["أ)", "ب)", "ج)", "د)"]
        raw = doc["choices"]
        choices = synthetic_parse_choices(raw, labels)
        choices = [re.sub(r"^\)?\s*", "", c).strip(" []'\"\n") for c in choices]
        latin_to_arabic = {"A": "أ", "B": "ب", "C": "ج", "D": "د"}
        arabic_to_latin = {v: k for k, v in latin_to_arabic.items()}
        valid_keys_arabic = list(latin_to_arabic.values())
        self_answer_arabic = doc["self_answer"].strip()
        self_answer_latin = arabic_to_latin.get(self_answer_arabic)        
        
        instruction = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n"
        question = doc["question"]
        query = f"{instruction}{question}\n"
        for arab_label, choice_text in zip(valid_keys_arabic, choices):
            choice_text = re.sub(r"^\)?\s*", "", choice_text).strip(" []'\"\n")
            query += f"{arab_label}. {choice_text}\n"
        query += "الإجابة:"

        return doc_to_text(query)
    return filter_fn

doc_to_text_biology = make_filter_synthetic("Biology")
doc_to_text_physics= make_filter_synthetic("Physics")
doc_to_text_math = make_filter_synthetic("Math")
doc_to_text_chemistry = make_filter_synthetic("Chemistry")
doc_to_text_general_science = make_filter_synthetic("General_Science")


