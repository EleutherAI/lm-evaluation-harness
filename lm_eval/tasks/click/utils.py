def get_context(doc) -> str:
    ctx = doc["paragraph"]
    q = doc["question"]
    opt = doc["options"]
    if ctx:
        res = f"주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.\n\n맥락: {ctx}\n질문: {q}\n보기:\nA:{opt[0]}, B: {opt[1]}, C: {opt[2]}, D: {opt[3]}\n정답:"
    else:
        res = f"주어진 질문을 천천히 읽고, 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.\n\n질문: {q}\n보기:\nA:{opt[0]}, B: {opt[1]}, C: {opt[2]}, D: {opt[3]}\n정답:"
    
    return res

def get_target(doc) -> str:
    ans = doc["answer"]
    return ['A', 'B', 'C', 'D'][doc["choices"].index(ans)]

def process_docs(dataset):
    def _process_doc(doc):
        cat = doc['id'].split('_')[1]