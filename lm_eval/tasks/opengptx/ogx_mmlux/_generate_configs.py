"""
Take in a YAML, and output all "other" splits with this YAML
"""

import json
import yaml
import argparse


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
    "BG": ("Въпрос", "Избори", "Отговор"),
    "DA": ("Spørgsmål", "Valgmuligheder", "Svar"),
    "DE": ("Frage", "Auswahlmöglichkeiten", "Antwort"),
    "ET": ("Küsimus", "Valikud", "Vastus"),
    "FI": ("Kysymys", "Valinnat", "Vastaa"),
    "FR": ("Question", "Choix", "Réponse"),
    "EL": ("Ερώτηση", "Επιλογές", "Απάντηση"),
    "IT": ("Domanda", "Scelte", "Risposta"),
    "LV": ("Jautājums", "Izvēle", "Atbilde"),
    "LT": ("Klausimas", "Pasirinkimai", "Atsakymas"),
    "NL": ("Vraag", "Keuzes", "Antwoord"),
    "PL": ("Pytanie", "Wybory", "Odpowiedź"),
    "PT-PT": ("Questão", "Escolhas", "Resposta"),
    "RO": ("Întrebare", "Alegeri", "Răspuns"),
    "SV": ("Fråga", "Valmöjligheter", "Svar"),
    "SK": ("Otázka", "Voľby", "Odpoveď"),
    "SL": ("Vprašanje", "Izbira", "Odgovor"),
    "ES": ("Pregunta", "Opciones", "Respuesta"),
    "CS": ("Otázka", "Volby", "Odpověď"),
    "HU": ("Kérdés", "Választások", "Válasz"),
}

CHOICES = {
    "BG": ("А", "Б", "В", "Г"),
    "DA": ("A", "B", "C", "D"),
    "DE": ("A", "B", "C", "D"),
    "ET": ("A", "B", "C", "D"),
    "FI": ("A", "B", "C", "D"),
    "FR": ("A", "B", "C", "D"),
    "EL": ("Α", "Β", "Γ", "Δ"),
    "IT": ("A", "B", "C", "D"),
    "LV": ("A", "B", "C", "D"),
    "LT": ("A", "B", "C", "D"),
    "NL": ("A", "B", "C", "D"),
    "PL": ("A", "B", "C", "D"),
    "PT-PT": ("A", "B", "C", "D"),
    "RO": ("A", "B", "C", "D"),
    "SV": ("A", "B", "C", "D"),
    "SK": ("A", "B", "C", "D"),
    "SL": ("A", "B", "C", "D"),
    "ES": ("A", "B", "C", "D"),
    "CS": ("A", "B", "C", "D"),
    "HU": ("A", "B", "C", "D"),
}

SUBJECTS = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--base_yaml", required=True)
    parser.add_argument("--descriptions", required=True)
    parser.add_argument("--prefix", default="ogx_mmlux")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    descriptions = json.load(open(args.descriptions, "r"))

    for lang in LANGS:
        _, _, answer = PROMPT_WORDS[lang]
        a, b, c, d = CHOICES[lang]

        for subj, cat in SUBJECTS.items():
            yaml_dict = {
                "include": args.base_yaml,
                "dataset_name": f"{subj}_{lang}",
                "task": f"{args.prefix}_{lang.lower()}-{subj}",
                # "task_alias": f"{subj}_{lang.lower()}",
                "group": f"{args.prefix}_{cat}",
                # "group_alias": f"{cat}",
                "doc_to_choice": f"['{a}', '{b}', '{c}', '{d}']",
                "doc_to_text": f"{{{{question.strip()}}}}\n{a}. {{{{choices[0]}}}}\n{b}. {{{{choices[1]}}}}\n{c}. {{{{choices[2]}}}}\n{d}. {{{{choices[3]}}}}\n{answer}:",
                "description": descriptions[lang][subj],
            }

            file_save_path = args.save_dir + f"{args.prefix}_{lang.lower()}-{subj}.yaml"

            with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(
                    yaml_dict,
                    yaml_file,
                    allow_unicode=True,
                    default_style='"',
                    sort_keys=False,
                )
