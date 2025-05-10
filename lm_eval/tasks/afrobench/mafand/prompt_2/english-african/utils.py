languages = {
    "amh": "Amharic",
    "bam": "Bambara",
    "bbj": "Gbomala",
    "ewe": "Ewe",
    "fon": "Fon",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "lug": "Luganda",
    "luo": "Luo",
    "mos": "Mossi",
    "nya": "Chichewa",
    "pcm": "Nigerian Pidgin",
    "sna": "Shona",
    "swa": "Swahili",
    "tsn": "Setswana",
    "twi": "Twi",
    "wol": "Wolof",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "Zulu",
}


def get_target(doc):
    target = (
        doc["translation"]["en"]
        if "en" in doc["translation"].keys()
        else doc["translation"]["fr"]
    )
    return target


def get_target_reverse(doc):
    target_key = [key for key in doc["translation"].keys() if key not in ["en", "fr"]][
        0
    ]
    target = doc["translation"][target_key]
    return target


def create_text_prompt_1(doc):
    source_key = [key for key in doc["translation"].keys() if key not in ["en", "fr"]][
        0
    ]
    source_sentence = doc["translation"][source_key]
    source_lang = "English" if "en" in doc["translation"].keys() else "French"
    prompt = (
        "You are an advanced Translator, a specialized assistant designed to translate documents from "
        f"{languages[source_key]} into {source_lang}. \nYour main goal is to ensure translations are grammatically "
        f"correct and human-oriented. \n{languages[source_key]}: {source_sentence} \n{source_lang}: "
    )
    return prompt


def create_reverse_prompt_1(doc):
    target_lang = [key for key in doc["translation"].keys() if key not in ["en", "fr"]][
        0
    ]
    source_key = "en" if "en" in doc["translation"].keys() else "fr"
    source_lang = "English" if source_key == "en" else "French"
    source_sentence = doc["translation"][source_key]
    prompt = (
        "You are an advanced Translator, a specialized assistant designed to translate documents from "
        f"{source_lang} into {languages[target_lang]}. \nYour main goal is to ensure translations are "
        f"grammatically correct and human-oriented. \n{source_lang}: {source_sentence} \n{languages[target_lang]}: "
    )
    return prompt


def create_text_prompt_2(doc):
    source_key = [key for key in doc["translation"].keys() if key not in ["en", "fr"]][
        0
    ]
    source_sentence = doc["translation"][source_key]
    source_lang = "English" if "en" in doc["translation"].keys() else "French"
    prompt = (
        f"{languages[source_key]} sentence: {source_sentence} \n{source_lang} sentence: ",
    )
    return prompt


def create_reverse_prompt_2(doc):
    target_lang = [key for key in doc["translation"].keys() if key not in ["en", "fr"]][
        0
    ]
    source_key = "en" if "en" in doc["translation"].keys() else "fr"
    source_lang = "English" if source_key == "en" else "French"
    source_sentence = doc["translation"][source_key]
    prompt = (
        f"{source_lang} sentence: {source_sentence} \n{languages[target_lang]} sentence: \n",
    )
    return prompt


def create_text_prompt_3(doc):
    source_key = [key for key in doc["translation"].keys() if key not in ["en", "fr"]][
        0
    ]
    source_sentence = doc["translation"][source_key]
    source_lang = "English" if "en" in doc["translation"].keys() else "French"
    prompt = (
        f"You are a translation expert. Translate the following {languages[source_key]} sentences "
        f"to {source_lang}. \n{languages[source_key]} sentence: {source_sentence}\n{source_lang} sentence: "
    )
    return prompt


def create_reverse_prompt_3(doc):
    target_lang = [key for key in doc["translation"].keys() if key not in ["en", "fr"]][
        0
    ]
    source_key = "en" if "en" in doc["translation"].keys() else "fr"
    source_lang = "English" if source_key == "en" else "French"
    source_sentence = doc["translation"][source_key]
    prompt = (
        f"You are a translation expert. Translate the following {source_lang} sentence into {languages[target_lang]}\n"
        f"{source_lang} sentence: {source_sentence}\n{languages[target_lang]} sentence: "
    )
    return prompt
