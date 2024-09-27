QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{{Question}}

A) {{A}}
B) {{B}}
C) {{C}}
D) {{D}}
""".strip()


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

LANGUAGES = {
    "AR_XY": "Arabic (Generic)",
    "BN_BD": "Bengali (Bangladesh)",
    "DE_DE": "German (Germany)",
    "ES_LA": "Spanish (Latin America)",
    "FR_FR": "French (France)",
    "HI_IN": "Hindi (India)",
    "ID_ID": "Indonesian (Indonesia)",
    "IT_IT": "Italian (Italy)",
    "JA_JP": "Japanese (Japan)",
    "KO_KR": "Korean (South Korea)",
    "PT_BR": "Portuguese (Brazil)",
    "ZH_CN": "Chinese (China)",
    "SW_KE": "Swahili (Kenya)",
    "YO_NG": "Yoruba (Nigeria)",
    "EN_US": "English (United States)",
}

ANSWERS = {
    "AR_XY": "الإجابة:",
    "BN_BD": "উত্তর:",
    "DE_DE": "Antwort:",
    "ES_LA": "Respuesta:",
    "FR_FR": "Réponse:",
    "HI_IN": "उत्तर:",
    "ID_ID": "Jawaban:",
    "IT_IT": "Risposta:",
    "JA_JP": "答え:",
    "KO_KR": "답변:",
    "PT_BR": "Resposta:",
    "ZH_CN": "回答:",
    "SW_KE": "Jawabu:",
    "YO_NG": "Idahun:",
    "EN_US": "Answer:",
}


def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )
