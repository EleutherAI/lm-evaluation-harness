import re


def doc_to_target(doc):
    choices = ["A", "B", "C", "D", "E"]
    try:
        return choices.index(doc["gt"])
    except:
        print(f"{doc}")
        print(f"No choices possible due to GT error")
        print(f"Force return 0")
        return 0


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


def build_tempate(doc, context):
    template = f"""
    ### 질문: {doc['question']}

      다음 각 선택지와 참고자료 를 읽고 A, B, C, D, E 중 하나를 선택하여 ''답변: A'' 와 같이 단답식으로 답해 주세요.

      ### 선택지:

      A. {doc['A']}

      B. {doc['B']}

      C. {doc['C']}

      D. {doc['D']}

      E. {doc['E']}

      ### 참고자료: 
      {context}

      ### 답변:
    """
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


def doc_to_text_q_rag_n8(doc):
    context = build_context_q(doc, n_max_context=8)
    return build_tempate(doc, context)


def doc_to_text_q_rag_n12(doc):
    context = build_context_q(doc, n_max_context=12)
    return build_tempate(doc, context)
