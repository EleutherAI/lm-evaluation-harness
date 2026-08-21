import ast

import numpy as np
from datasets import Features, Sequence, Value, load_dataset


DATASET_REPO = "gplsi/cieacova"
CHOICE_CONFIG = "multiple_choice"
GEN_CONFIG = "text_generation"

_FEWSHOT_CHOICE_BY_INSTRUCTION = None
_FEWSHOT_GEN_BY_INSTRUCTION = None

_CIEACOVA_FEATURES = Features(
    {
        "Tarea": Value("string"),
        "Instrucción": Value("string"),
        "Pista": Value("string"),
        "Pregunta": Value("string"),
        "Respuesta": Sequence(Value("string")),
        "Opciones": Sequence(Value("string")),
        "Metadata": {
            "Anotador": Value("string"),
            "Año": Value("string"),
            "Dificultad": Value("string"),
            "Fuente": Value("string"),
            "Idioma": Value("string"),
            "Mes": Value("string"),
            "N. Pregunta": Value("string"),
            "Sección": Value("string"),
        },
        "Prompt": Value("string"),
        "Subtarea": Value("string"),
    }
)


def load_cieacova_dataset(**kwargs):
    dataset_repo = kwargs.pop("dataset_repo", DATASET_REPO)
    dataset_name = kwargs.pop("dataset_name", None)
    kwargs.pop("config_source", None)
    if not dataset_name:
        raise ValueError("dataset_name must be specified in dataset_kwargs")

    allowed_kwargs = {
        "cache_dir",
        "data_dir",
        "data_files",
        "download_config",
        "download_mode",
        "keep_in_memory",
        "num_proc",
        "revision",
        "save_infos",
        "split",
        "storage_options",
        "streaming",
        "token",
        "verification_mode",
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}

    # Force a stable schema across train/test to avoid null-vs-string cast failures.
    return load_dataset(
        dataset_repo,
        dataset_name,
        features=_CIEACOVA_FEATURES,
        **filtered_kwargs,
    )


def _parse_list_like(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        try:
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (SyntaxError, ValueError):
            pass
        return [
            part.strip().strip("'\"")
            for part in cleaned.strip("[]").split(",")
            if part.strip()
        ]
    return [str(value).strip()]


def _normalize_instruction(text):
    if not text:
        return ""
    return text.replace('"[MASK]"', "[MASK]")


def _format_option_lines(options):
    if not options:
        return ""
    return "\n".join(f"- {opt}" for opt in options)


def _build_choice_fewshot_map():
    train = load_cieacova_dataset(dataset_name=CHOICE_CONFIG, split="train")
    mapping = {}
    for row in train:
        instruction = _normalize_instruction(row.get("Instrucción", ""))
        if not instruction or instruction in mapping:
            continue
        options = _parse_list_like(row.get("Opciones"))
        answers = _parse_list_like(row.get("Respuesta"))
        mapping[instruction] = {
            "Pregunta": row.get("Pregunta", ""),
            "Opciones": options,
            "Respuesta": answers[0] if answers else "",
        }
    return mapping


def _build_gen_fewshot_map():
    train = load_cieacova_dataset(dataset_name=GEN_CONFIG, split="train")
    mapping = {}
    for row in train:
        instruction = _normalize_instruction(row.get("Instrucción", ""))
        if not instruction or instruction in mapping:
            continue
        answers = _parse_list_like(row.get("Respuesta"))
        mapping[instruction] = {
            "Pregunta": row.get("Pregunta", ""),
            "Pista": row.get("Pista", ""),
            "Respuesta": answers[0] if answers else "",
        }
    return mapping


def _get_choice_fewshot_map():
    global _FEWSHOT_CHOICE_BY_INSTRUCTION
    if _FEWSHOT_CHOICE_BY_INSTRUCTION is None:
        _FEWSHOT_CHOICE_BY_INSTRUCTION = _build_choice_fewshot_map()
    return _FEWSHOT_CHOICE_BY_INSTRUCTION


def _get_gen_fewshot_map():
    global _FEWSHOT_GEN_BY_INSTRUCTION
    if _FEWSHOT_GEN_BY_INSTRUCTION is None:
        _FEWSHOT_GEN_BY_INSTRUCTION = _build_gen_fewshot_map()
    return _FEWSHOT_GEN_BY_INSTRUCTION


def process_docs(dataset):
    fewshot_by_instruction = _get_choice_fewshot_map()

    def _process_doc(doc):
        instruction = _normalize_instruction(doc.get("Instrucción", ""))
        options = _parse_list_like(doc.get("Opciones"))
        answers = _parse_list_like(doc.get("Respuesta"))

        correct_indices = [
            options.index(answer) for answer in answers if answer in options
        ]
        if not correct_indices:
            correct_indices = [0]

        fewshot = fewshot_by_instruction.get(instruction)
        if fewshot:
            option_lines = _format_option_lines(fewshot["Opciones"])
            doc["dynamic_fewshot"] = (
                f"Exemple de text a resoldre:\n{fewshot['Pregunta']}\n\n"
                f"Exemple opcions de resposta:\n{option_lines}\n\n"
                f"Exemple resposta:\n{fewshot['Respuesta']}\n\n"
                "---\n\n"
            )
        else:
            doc["dynamic_fewshot"] = ""

        doc["Instrucción"] = instruction
        doc["parsed_opciones"] = options or [""]
        doc["parsed_opciones_text"] = _format_option_lines(doc["parsed_opciones"])
        doc["parsed_respuesta"] = correct_indices
        return doc

    return dataset.map(_process_doc)


def process_results(doc, results):
    lls = [res[0] if isinstance(res, tuple) else res for res in results]
    options = doc["parsed_opciones"]
    targets = doc["parsed_respuesta"]

    lls_norm = [ll / max(1, len(opt)) for ll, opt in zip(lls, options)]
    pred = int(np.argmax(lls))
    pred_norm = int(np.argmax(lls_norm))

    acc = 1.0 if pred in targets else 0.0
    acc_norm = 1.0 if pred_norm in targets else 0.0

    difficulty = doc.get("Metadata", {}).get("Dificultad", "Desconocida")
    subtask = doc["Subtarea"]
    return {
        "acc": acc,
        "acc_norm": acc_norm,
        f"acc_{difficulty}": acc,
        f"acc_norm_{difficulty}": acc_norm,
        f"acc_{subtask}": acc,
        f"acc_norm_{subtask}": acc_norm,
    }


def process_gen_docs(dataset):
    fewshot_by_instruction = _get_gen_fewshot_map()

    def _process_doc(doc):
        instruction = _normalize_instruction(doc.get("Instrucción", ""))
        answers = _parse_list_like(doc.get("Respuesta"))
        doc["parsed_respuesta"] = answers or [""]

        fewshot = fewshot_by_instruction.get(instruction)
        if fewshot:
            hint = fewshot.get("Pista", "")
            hint_section = f"Exemple PISTA:\n{hint}\n\n" if hint else ""
            doc["dynamic_fewshot"] = (
                f"Exemple de text a resoldre:\n{fewshot.get('Pregunta', '')}\n\n"
                f"{hint_section}"
                f"Exemple resposta:\n{fewshot.get('Respuesta', '')}\n\n"
                "---\n\n"
            )
        else:
            doc["dynamic_fewshot"] = ""

        doc["Instrucción"] = instruction
        return doc

    return dataset.map(_process_doc)


def clean_text(text, allowed_specials):
    if not text:
        return ""
    return "".join(
        c.lower() for c in text if c.isalnum() or c in allowed_specials
    ).strip()


def process_results_text_generation(doc, results):
    pred = results[0] if isinstance(results, list) else results
    targets = doc["parsed_respuesta"]

    allowed_specials = {
        char
        for target in targets
        for char in target
        if not char.isalnum() and not char.isspace()
    }
    pred_clean = clean_text(pred, allowed_specials)
    targets_clean = [clean_text(target, allowed_specials) for target in targets]

    acc = 1.0 if pred_clean in targets_clean else 0.0
    difficulty = doc.get("Metadata", {}).get("Dificultad", "Desconocida")
    subtask = doc["Subtarea"]
    return {
        "exact_match": acc,
        f"exact_match_{difficulty}": acc,
        f"exact_match_{subtask}": acc,
    }
