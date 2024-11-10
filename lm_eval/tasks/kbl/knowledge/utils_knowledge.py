def build_context_q(doc, n_max_context=3):
    context = ""
    for i, r1 in enumerate(
        doc["rag_data"]["question_only"]["retrieved_docs"]["results"]
    ):
        if i >= n_max_context:
            break
        context += f"\n참고문헌 {i+1}: {r1['title']}\n{r1['passage']}"

    context = context.strip()
    return context


def build_context_qa(doc, n_max_context=3):
    context = ""
    for i, r1 in enumerate(
        doc["rag_data"]["question_and_answers"]["retrieved_docs"]["results"]
    ):
        if i >= n_max_context:
            break
        context += f"\n참고문헌 {i+1}: {r1['title']}\n{r1['passage']}"

    context = context.strip()
    return context


def build_tempate(doc, context):
    selection_keys = []
    assert "G" not in doc
    if "A" in doc:
        selection_keys.append("A")
    if "B" in doc:
        selection_keys.append("B")
    if "C" in doc:
        selection_keys.append("C")
    if "D" in doc:
        selection_keys.append("D")
    if "E" in doc:
        selection_keys.append("E")
    if "F" in doc:
        selection_keys.append("F")

    template = (
        f"### 질문: {doc['question']}"
        f"\n다음 각 선택지와 참고문헌 를 읽고 {', '.join(selection_keys)} 중 하나를 선택하여 ''답변: A'' 와 같이 단답식으로 답해 주세요."
        f"\n### 선택지:"
    )
    if "A" in doc:
        template += f"\nA. {doc['A']}"
    if "B" in doc:
        template += f"\nB. {doc['B']}"
    if "C" in doc:
        template += f"\nC. {doc['C']}"
    if "D" in doc:
        template += f"\nD. {doc['D']}"
    if "E" in doc:
        template += f"\nE. {doc['E']}"
    if "F" in doc:
        template += f"\nF. {doc['F']}"

    template += f"\n### 참고문헌: {context}"
    template += f"\n### 답변:"

    return template


def doc_to_text_q_rag_n1(doc):
    context = build_context_q(doc, n_max_context=1)
    return build_tempate(doc, context)


def doc_to_text_q_rag_n2(doc):
    context = build_context_q(doc, n_max_context=2)
    return build_tempate(doc, context)


def doc_to_text_q_rag_n3(doc):
    context = build_context_q(doc, n_max_context=3)
    return build_tempate(doc, context)


def doc_to_text_q_rag_n4(doc):
    context = build_context_q(doc, n_max_context=4)
    return build_tempate(doc, context)


def doc_to_text_q_rag_n6(doc):
    context = build_context_q(doc, n_max_context=6)
    return build_tempate(doc, context)


def doc_to_text_q_rag_n8(doc):
    context = build_context_q(doc, n_max_context=8)
    return build_tempate(doc, context)


def doc_to_text_q_rag_n12(doc):
    context = build_context_q(doc, n_max_context=12)
    return build_tempate(doc, context)


def doc_to_text_qa2_rag(doc):
    template = build_template_qa2(doc, n_max_context_per_selection=2)
    return template


def doc_to_text_qa2_rag_n1(doc):
    template = build_template_qa2(doc, n_max_context_per_selection=1)
    return template


def doc_to_text_qa2_rag_n2(doc):
    template = build_template_qa2(doc, n_max_context_per_selection=2)
    return template


def doc_to_text_qa2_rag_n4(doc):
    template = build_template_qa2(doc, n_max_context_per_selection=4)
    return template


def doc_to_text_qa2_rag_n8(doc):
    template = build_template_qa2(doc, n_max_context_per_selection=8)
    return template


def doc_to_text_qa2_rag_shortest_doc(doc):
    template = build_template_qa2(doc, n_max_context_per_selection=-1)
    return template


def build_template_qa2(doc, n_max_context_per_selection=1):
    # return template
    rag_data = doc["rag_data"]
    if rag_data["question_only"] is not None:
        assert rag_data["question_and_individual_answers"] is None
        context = build_context_q(doc, 5 * n_max_context_per_selection)
        template = build_tempate(doc, context)
    else:
        template = (
            f"### 질문: {doc['question']}"
            "\n다음 각 선택지와 선택지별 참고문헌을 읽고 A, B, C, D 중 하나를 선택하여 ''답변: A'' 와 같이 단답식으로 답해 주세요."
            f"\nA. {doc['A']}\n{build_context_qa2(doc, 'A', n_max_context_per_selection)}"
            f"\nB. {doc['B']}\n{build_context_qa2(doc, 'B', n_max_context_per_selection)}"
            f"\nC. {doc['C']}\n{build_context_qa2(doc, 'C', n_max_context_per_selection)}"
            f"\nD. {doc['D']}\n{build_context_qa2(doc, 'D', n_max_context_per_selection)}"
        )
    return template


def build_context_qa2(doc, key, n_max_context=2):
    """
    key : A, B, C, D
    """

    context = ""
    if n_max_context < 0:
        results = doc["rag_data"]["question_and_individual_answers"][key][
            "retrieved_docs"
        ]["results"]
        ls = [len(r["passage"]) for r in results[0:3]]
        idx = None
        score = 1e10
        for i, l in enumerate(ls):
            if l < score:
                score = l
                idx = i
        r1 = results[idx]
        context += f"\n참고문헌 {key}{1}: {r1['title']}, {r1['passage']}"
    else:
        for i, r1 in enumerate(
            doc["rag_data"]["question_and_individual_answers"][key]["retrieved_docs"][
                "results"
            ]
        ):
            if i >= n_max_context:
                break
            context += f"\n참고문헌 {key}{i+1}: {r1['title']}, {r1['passage']}"

    context = context.strip()
    return context


def doc_to_choice_qa2_rag_n1(doc):
    keys = ["A", "B", "C", "D", "E"]
    choices = []
    for key in keys:
        answer = ""
        if key in doc:
            ref = build_context_qa2(doc, key, n_max_context=1)
            answer += f"### 참고문헌들\n{ref}\n### 답변: {doc[key]}"
            choices.append(answer)

    return choices


def doc_to_choice_qa2_rag_n2(doc):
    keys = ["A", "B", "C", "D", "E"]
    choices = []
    for key in keys:
        answer = ""
        if key in doc:
            ref = build_context_qa2(doc, key, n_max_context=1)
            answer += f"### 참고문헌들\n{ref}\n### 답변: {doc[key]}"
            choices.append(answer)

    return choices


def build_context_qa2(doc, key, n_max_context=3):
    """
    key : A, B, C, D, E
    """
    context = ""
    if n_max_context < 0:
        results = doc["rag_data"]["question_and_individual_answers"][key][
            "retrieved_docs"
        ]["results"]
        ls = [len(r["passage"]) for r in results[0:13]]
        idx = None
        score = 1e10
        for i, l in enumerate(ls):
            if l < score:
                score = l
                idx = i
        r1 = results[idx]
        context += f"\n참고문헌 {key}{1}: {r1['title']}, {r1['passage']}"
    else:
        for i, r1 in enumerate(
            doc["rag_data"]["question_and_individual_answers"][key]["retrieved_docs"][
                "results"
            ]
        ):
            if i >= n_max_context:
                break
            context += f"\n참고문헌 {key}{i+1}: {r1['title']}, {r1['passage']}"

        context = context.strip()
    return context


def doc_to_choice_q_rag_n1(doc):
    keys = ["A", "B", "C", "D", "E"]
    choices = []
    ref = build_context_q(doc, n_max_context=1)
    for key in keys:
        answer = ""
        if key in doc:
            answer += f"### 참고문헌들\n{ref}\n### 답변: {doc[key]}"
            choices.append(answer)

    return choices
