import os
import yaml
import argparse

from lm_eval.utils import logging


LANGS = [
    "BG",
    "DA",
    "DE",
    "ET",
    "FI",
    "FR",
    "EL",
    "IT",
    "LV",
    "LT",
    "NL",
    "PL",
    "PT-PT",
    "RO",
    "SV",
    "SK",
    "SL",
    "ES",
    "CS",
    "HU",
]


PROMPT_WORDS = {
    "BG": ("Въпрос", "Отговор"),
    "DA": ("Spørgsmål", "Svar"),
    "DE": ("Frage", "Antwort"),
    "ET": ("Küsimus", "Vastus"),
    "FI": ("Kysymys", "Vastaa"),
    "FR": ("Question", "Réponse"),
    "EL": ("Ερώτηση", "Απάντηση"),
    "IT": ("Domanda", "Risposta"),
    "LV": ("Jautājums", "Atbilde"),
    "LT": ("Klausimas", "Atsakymas"),
    "NL": ("Vraag", "Antwoord"),
    "PL": ("Pytanie", "Odpowiedź"),
    "PT-PT": ("Questão", "Resposta"),
    "RO": ("Întrebare", "Răspuns"),
    "SV": ("Fråga", "Svar"),
    "SK": ("Otázka", "Odpoveď"),
    "SL": ("Vprašanje", "Odgovor"),
    "ES": ("Pregunta", "Respuesta"),
    "CS": ("Otázka", "Odpověď"),
    "HU": ("Kérdés", "Válasz"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="ogx_arcx")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]

    for split in ["easy", "challenge"]:
        for lang in LANGS:
            yaml_dict = {
                "include": base_yaml_name,
                "task": f"ogx_arcx_{split}_{lang.lower()}",
                "dataset_name": f"{split}_{lang}",
                "doc_to_text": f"{PROMPT_WORDS[lang][0]}: {{{{question}}}}\n{PROMPT_WORDS[lang][1]}:",
                "doc_to_decontamination_query": f"{PROMPT_WORDS[lang][0]}: {{{{question}}}}\n{PROMPT_WORDS[lang][1]}:",
            }

            file_save_path = f"{args.save_prefix_path}_{split}_{lang.lower()}.yaml"

            logging.info(f"Saving yaml for subset {split}_{lang} to {file_save_path}")

            with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(
                    yaml_dict,
                    yaml_file,
                    allow_unicode=True,
                    default_style='"',
                    sort_keys=False,
                )
