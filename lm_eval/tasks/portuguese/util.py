from datasets import load_dataset, Dataset, DatasetDict

LETTERS = ["A", "B", "C", "D", "E"]

BROVERBS_PROVERB_TO_HISTORY_FEWSHOT_IDS = [419, 394, 293, 349, 264]
BROVERBS_HISTORY_TO_PROVERB_FEWSHOT_IDS = [493, 462, 345, 217, 404]
ENEM_FEWSHOT_IDS = ["2022_21", "2022_88", "2022_143"]
BLUEX_FEWSHOT_IDS = ["USP_2018_3", "UNICAMP_2018_2", "USP_2018_35", "UNICAMP_2018_16", "USP_2018_89"]

def broverbs_correct_index_target(doc):
    return LETTERS[doc["correct_index"]]

def broverbs_proverb_to_history_doc_to_text(doc):
    prompt = (
        "A seguir há um provérbio brasileiro. "
        "Escolha a história que melhor corresponde ao seu significado.\n\n"
        f"Provérbio:\n{doc['input']}\n\n"
        "Alternativas:\n"
    )

    for i, alt in enumerate(doc["alternatives"]):
        prompt += f"{LETTERS[i]}. {alt}\n"

    prompt += "\nResposta correta:"
    return prompt


def broverbs_proverb_to_history_custom_dataset(**kwargs):
    ds = load_dataset("Tropic-AI/BRoverbs")
    original = ds["proverb_to_history"]
    fewshot_ids = set(BROVERBS_PROVERB_TO_HISTORY_FEWSHOT_IDS)

    filtered_docs = []
    for row_id, doc in enumerate(original):
        if row_id in fewshot_ids:
            continue

        new_doc = dict(doc)
        new_doc["row_id"] = row_id
        filtered_docs.append(new_doc)

    return DatasetDict({
        "test": Dataset.from_list(filtered_docs)
    })

def broverbs_proverb_to_history_fewshot_samples():
    ds = load_dataset("Tropic-AI/BRoverbs", split="proverb_to_history")

    samples = [ds[row_id] for row_id in BROVERBS_PROVERB_TO_HISTORY_FEWSHOT_IDS]

    if len(samples) != len(BROVERBS_PROVERB_TO_HISTORY_FEWSHOT_IDS):
        raise ValueError(
            "Não foi possível montar os few-shots de proverb_to_history do BRoverbs."
        )

    return samples

def broverbs_history_to_proverb_doc_to_text(doc):
    prompt = (
        "A seguir há uma história. "
        "Escolha o provérbio brasileiro que melhor corresponde ao seu significado.\n\n"
        f"História:\n{doc['input']}\n\n"
        "Alternativas:\n"
    )

    for i, alt in enumerate(doc["alternatives"]):
        prompt += f"{LETTERS[i]}. {alt}\n"

    prompt += "\nResposta correta:"
    return prompt

def broverbs_history_to_proverb_fewshot_samples():
    ds = load_dataset("Tropic-AI/BRoverbs", split="history_to_proverb")
    samples =  [ds[row_id] for row_id in BROVERBS_HISTORY_TO_PROVERB_FEWSHOT_IDS]

    if len(samples) != len(BROVERBS_HISTORY_TO_PROVERB_FEWSHOT_IDS):
        raise ValueError(
            "Não foi possível montar os few-shots de history_to_proverb do BRoverbs."
        )

    return samples

def broverbs_history_to_proverb_custom_dataset(**kwargs):
    ds = load_dataset("Tropic-AI/BRoverbs")
    original = ds["history_to_proverb"]
    fewshot_ids = set(BROVERBS_HISTORY_TO_PROVERB_FEWSHOT_IDS)

    filtered_docs = []
    balanced_idx = 0

    for row_id, doc in enumerate(original):
        if row_id in fewshot_ids:
            continue

        target_index = balanced_idx % 5
        new_doc = broverbs_rebalance_doc(doc, target_index)
        new_doc["row_id"] = row_id

        filtered_docs.append(new_doc)
        balanced_idx += 1

    return DatasetDict({
        "test": Dataset.from_list(filtered_docs)
    })

def broverbs_rebalance_doc(doc, target_index):
    alternatives = list(doc["alternatives"])
    correct_index = doc["correct_index"]

    correct_alt = alternatives[correct_index]
    wrong_alts = [alt for i, alt in enumerate(alternatives) if i != correct_index]

    new_alts = wrong_alts[:]
    new_alts.insert(target_index, correct_alt)

    new_doc = dict(doc)
    new_doc["alternatives"] = new_alts
    new_doc["correct_index"] = target_index
    return new_doc

def enem_generate_options(choices):
    options = ""
    for text, label in zip(choices["text"], choices["label"]):
        options += f"{label}. {text}\n"
    return options.strip()

def enem_doc_to_text(doc):
    return (
        f"Pergunta:\n{doc['question']}\n"
        f"Alternativas:\n{enem_generate_options(doc['choices'])}\n"
        f"Resposta correta:"
    )

def enem_fewshot_samples():
    ds = load_dataset("eduagarcia/enem_challenge", split="train")

    selected_by_id = {row["id"]: row for row in ds if row["id"] in set(ENEM_FEWSHOT_IDS)}
    samples = [selected_by_id[_id] for _id in ENEM_FEWSHOT_IDS if _id in selected_by_id]

    if len(samples) != len(ENEM_FEWSHOT_IDS):
        missing = [x for x in ENEM_FEWSHOT_IDS if x not in selected_by_id]
        raise ValueError(f"Few-shot IDs não encontrados no dataset: {missing}")

    return samples

def enem_custom_dataset(**kwargs):
    ds = load_dataset("eduagarcia/enem_challenge")
    original = ds["train"]
    fewshot_ids = set(ENEM_FEWSHOT_IDS)

    filtered_docs = []
    for doc in original:
        if doc["id"] in fewshot_ids:
            continue

        new_doc = dict(doc)
        filtered_docs.append(new_doc)

    return DatasetDict({
        "test": Dataset.from_list(filtered_docs)
    })

def bluex_fewshot_samples():
    ds = load_dataset("eduagarcia-temp/BLUEX_without_images", split="train")

    selected_by_id = {row["id"]: row for row in ds if row["id"] in set(BLUEX_FEWSHOT_IDS)}
    samples = [selected_by_id[_id] for _id in BLUEX_FEWSHOT_IDS if _id in selected_by_id]

    if len(samples) != len(BLUEX_FEWSHOT_IDS):
        missing = [x for x in BLUEX_FEWSHOT_IDS if x not in selected_by_id]
        raise ValueError(f"Few-shot IDs não encontrados no dataset: {missing}")

    return samples

def bluex_custom_dataset(**kwargs):
    ds = load_dataset("eduagarcia-temp/BLUEX_without_images")
    original = ds["train"]
    fewshot_ids = set(BLUEX_FEWSHOT_IDS)

    filtered_docs = []
    for doc in original:
        if doc["id"] in fewshot_ids:
            continue

        new_doc = dict(doc)
        filtered_docs.append(new_doc)

    return DatasetDict({
        "test": Dataset.from_list(filtered_docs)
    })

def portuguese_hate_speech_binary_fewshot_samples():
    ds = load_dataset(
        "eduagarcia/portuguese_benchmark",
        "Portuguese_Hate_Speech_binary",
        split="train",
    )

    wanted_ids = [
        52, 50, 39, 28, 3, 105, 22, 25, 60, 11,
        66, 41, 9, 4, 91, 42, 7, 20, 76, 1,
        104, 13, 67, 54, 97, 27, 24, 14, 16, 48,
        53, 40, 34, 49, 32, 119, 114, 2, 58, 83,
        18, 36, 5, 6, 10, 35, 38, 0, 21, 46
    ]

    selected_by_id = {row["idx"]: row for row in ds if row["idx"] in set(wanted_ids)}
    samples = [selected_by_id[_id] for _id in wanted_ids if _id in selected_by_id]

    if len(samples) != len(wanted_ids):
        missing = [x for x in wanted_ids if x not in selected_by_id]
        raise ValueError(f"Few-shot IDs não encontrados no dataset: {missing}")

    return samples

def assin2_float_to_pt_str(doc):
    return "{:.1f}".format(doc['relatedness_score']).replace('.', ',')

sparrow_emotion_por_labels = ['Admiration', 'Amusement', 'Anger', 'Annoyance', 'Approval', 'Compassion', 'Confusion', 'Curiosity', 'Desire', 'Disappointment', 'Disapproval', 'Disgust', 'Embarrassment', 'Envy', 'Excitement', 'Fear', 'Gratitude', 'Grief', 'Joy', 'Longing', 'Love', 'Nervousness', 'Optimism', 'Pride', 'Relief', 'Remorse', 'Sadness', 'Surprise']
sparrow_emotion_por_trans = ['Admiração', 'Diversão', 'Raiva', 'Aborrecimento', 'Aprovação', 'Compaixão', 'Confusão', 'Curiosidade', 'Desejo', 'Decepção', 'Desaprovação', 'Nojo', 'Vergonha', 'Inveja', 'Entusiasmo', 'Medo', 'Gratidão', 'Luto', 'Alegria', 'Saudade', 'Amor', 'Nervosismo', 'Otimismo', 'Orgulho', 'Alívio' , 'Remorso', 'Tristeza', 'Surpresa']

def sparrow_emotion_por_trans_label(doc):
    return sparrow_emotion_por_trans[sparrow_emotion_por_labels.index(doc['label'])]
