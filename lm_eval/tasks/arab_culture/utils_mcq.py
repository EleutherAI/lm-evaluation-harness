import os

from lm_eval.tasks.arab_culture.prompts import (
    BASE_PROMPT,
    BASE_PROMPT_AR,
    JAIS_CHAT_AR,
    JAIS_CHAT_EN,
    REGION_COUNTRY_PROMPT,
    REGION_COUNTRY_PROMPT_AR,
    REGION_PROMPT,
    REGION_PROMPT_AR,
)


### get the conutry variable from environment

### Set this to one to add the country and region information to the prompt
COUNTRY = True if os.getenv("COUNTRY", True) == "True" else False
### Set this to one to add the region information to the prompt
REGION = True if os.getenv("REGION", True) == "True" else False
### Set this to change between Arabic and English for the answer keys and the choices keys
ARABIC = True if os.getenv("ARABIC", True) == "True" else False
### Get the model name
MODEL_NAME = os.getenv("MODEL_NAME")
## Uncomment this to check if the environment variables are set correctly
# print(f'Task settings: COUNTRY: {COUNTRY}, REGION: {REGION}, ARABIC: {ARABIC}', MODEL_NAME: {MODEL_NAME})

en_ar_countries_regions = {
    "Egypt": "مصر",
    "Morocco": "المغرب",
    "Algeria": "الجزائر",
    "Libya": "ليبيا",
    "Sudan": "السودان",
    "Tunisia": "تونس",
    "Jordan": "الأردن",
    "Lebanon": "لبنان",
    "Syria": "سوريا",
    "Palestine": "فلسطين",
    "Yemen": "اليمن",
    "UAE": "الإمارات",
    "KSA": "السعودية",
    "Gulf": "الخليج",
    "Levant": "الشام",
    "North Africa": "شمال أفريقيا",
    "Nile Valley": "وادي النيل",
}


def doc_to_text(doc):
    country = "" if not doc["country"] else doc["country"]
    region = "" if not doc["region"] else doc["region"]
    first_statement = doc["first_statement"].strip()

    ## We don't have a setting for only information about the country without the region
    if COUNTRY:
        assert REGION, (
            "If you want to add the country information, you must also add the region information"
        )

    ## convert contry and region name to arabic if the language is arabic
    if ARABIC:
        country = en_ar_countries_regions[country]
        region = en_ar_countries_regions[region]

    choices = doc["options"]
    choices_str = ""
    for i in range(3):
        key = choices["arabic_keys"][i] if ARABIC else choices["english_keys"][i]
        choice_str = key + ". " + choices["text"][i].strip() + "\n"
        choices_str += choice_str

    if COUNTRY and REGION:
        cur_prompt = REGION_COUNTRY_PROMPT_AR if ARABIC else REGION_COUNTRY_PROMPT
        doc_text = cur_prompt.format(
            country=country,
            region=region,
            first_statement=first_statement,
            choices=choices_str,
        )
    elif REGION:
        cur_prompt = REGION_PROMPT_AR if ARABIC else REGION_PROMPT
        doc_text = cur_prompt.format(
            region=region, first_statement=first_statement, choices=choices_str
        )
    else:
        cur_prompt = BASE_PROMPT_AR if ARABIC else BASE_PROMPT
        doc_text = cur_prompt.format(
            first_statement=first_statement, choices=choices_str
        )

    ### apply jais chat template
    if MODEL_NAME and "jais" in MODEL_NAME and "chat" in MODEL_NAME:
        if ARABIC:
            doc_text = JAIS_CHAT_AR.format(question=doc_text)
        else:
            doc_text = JAIS_CHAT_EN.format(question=doc_text)

    return doc_text


def doc_to_choice(doc):
    return doc["options"]["arabic_keys"] if ARABIC else doc["options"]["english_keys"]


def doc_to_target(doc):
    ans = (
        doc["answer_key"]["arabic_answer_key"]
        if ARABIC
        else doc["answer_key"]["english_answer_key"]
    )
    ans = ans.strip()
    return ans
