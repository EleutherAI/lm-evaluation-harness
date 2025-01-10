PROMPT = "This is a {}. Select the correct answer!\n\nQuestion: {}\n{}\n\nAnswer:"

level_en = {
    "Primary": "primary school",
    "Middle": "middle school",
    "High": "high school",
    "Univ": "university",
    "Prof": "professional",
}

alpa = ["A.", "B.", "C.", "D.", "E."]


def doc_to_text(doc):
    """
    Refactoring `prepare_data_en` to fit with the lm harness framework.
    https://github.com/mbzuai-nlp/ArabicMMLU/blob/main/util_prompt.py
    """

    level = "" if not doc["Level"] else " for " + level_en[doc["Level"]]
    country = "" if not doc["Country"] else " in " + doc["Country"]
    main_meta_data = f"{doc['Subject']} question{level}{country}"

    question = (
        doc["Question"]
        if doc["Context"] == ""
        else f"{doc['Context']}\n\n{doc['Question']}"
    )

    options = []
    for i, opt in enumerate(
        ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]
    ):
        if not doc[opt]:
            break
        options.append(f"{alpa[i]} {doc[opt]}")

    doc_text = PROMPT.format(main_meta_data, question, "\n".join(options))

    return doc_text


def doc_to_choice(doc):
    return [alpa[i][0] for i in range(5) if doc[f"Option {i+1}"]]
