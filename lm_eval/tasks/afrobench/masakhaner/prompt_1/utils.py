from lm_eval.utils import weighted_f1_score


def doc_to_target(doc):
    return transform_text(doc['ner_tags'])


def transform_text(text):
    entities = []
    current_entity = ""
    current_tag = ""

    for pair in text.split('\n'):
        if pair:  # Check if the line is not empty
            word, tag = pair.strip().split(': ')
            tag = tag.upper()
            word = word.lower()
            word = word.strip(',.').strip()

            if tag.startswith('B-'):
                if current_entity:
                    entities.append(f"{current_tag}: {current_entity}")
                current_tag = tag.split('-')[1]
                current_entity = word
            elif tag.startswith('I-') and tag.split('-')[1] == current_tag:
                current_entity += word
            else:
                if current_entity:
                    entities.append(f"{current_tag}: {current_entity}")
                    current_entity = ""
                    current_tag = ""
    if current_entity:
        entities.append(f"{current_tag}: {current_entity}")

        # Join all the transformed output lines with $$ as separator
    return ' $$ '.join(entities)
