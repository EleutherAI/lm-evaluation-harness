def doc_to_text(doc) -> str:
    options = doc["options"] + [""] * (5 - len(doc["options"])) 
    choices = "\n".join(f"{i+1}. {v}" for i, v in enumerate(options))
    return (
        f"다음은 {doc['license_name']} 자격시험 기출문제입니다.\n"
        "문제를 잘 읽고, 보기 중에서 가장 적절한 정답을 고르시오.\n"
        f"{choices}\n정답:"
    )

def doc_to_target(doc) -> int:
    return int(doc["solution"])

def doc_to_choice(doc):
    return doc["options"] + [""] * (5 - len(doc["options"]))