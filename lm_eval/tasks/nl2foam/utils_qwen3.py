import numpy as np
from transformers import AutoTokenizer

try:
    import evaluate

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError(
        "Please install evaluation metrics via pip install evaluate bert-score "
        "rouge_score>=0.1.2 nltk absl-py "
        "git+https://github.com/google-research/bleurt.git"
    )
except Exception as e:
    raise RuntimeError(
        f"Error loading evaluation metrics: {str(e)}. Please check your installation."
    )


TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

def doc_to_text_formatted(doc):
    messages = doc.get("prompt", [])

    assert messages[0]["role"] == "system", "System message is not the first message."
    assert messages[1]["role"] == "user", "User message is not the second message."

    # replace in system message and assistant message's content tags <reasoning> </reasoning> with <think> and </think>
    messages[0]["content"] = messages[0]["content"].replace("<reasoning>", "<think>").replace("</reasoning>", "</think>")

    message_text = TOKENIZER.apply_chat_template(  # type: ignore
        conversation=messages,
        tokenize=False
    )

    return message_text


def doc_to_target_formatted(doc):
    messages = doc.get("completion", [])
    assert messages[0]["role"] == "assistant", "Assistant message is not the first message."

    message_text = TOKENIZER.apply_chat_template(
        conversation=messages,
        tokenize=False
    )
    # replace in assistant message's content tags <reasoning> </reasoning> with <think> and </think>
    message_text = message_text.replace("<reasoning>", "<think>").replace("</reasoning>", "</think>")

    return message_text


def doc_to_text_original(doc):
    instruction = doc.get("instruction", "")
    user_input = doc.get("input", "")

    messages = []
    if instruction:
        messages.append({"role": "system", "content": instruction})
    if user_input:
        messages.append({"role": "user", "content": user_input})

    assert len(messages) == 2, "There should be 2 messages."
    assert messages[0]["role"] == "system", "System message is not the first message."
    assert messages[1]["role"] == "user", "User message is not the second message."

    message_text = TOKENIZER.apply_chat_template(  # type: ignore
        conversation=messages,
        tokenize=False
    )

    return message_text


def doc_to_target_original(doc):
    output = doc.get("output", "")

    # Construct messages
    messages = []
    if output:
        messages.append({"role": "assistant", "content": output})

    assert len(messages) == 1, "There should be 1 message."
    assert messages[0]["role"] == "assistant", "Assistant message is not the first message."

    message_text = TOKENIZER.apply_chat_template(
        conversation=messages,
        tokenize=False
    )

    return message_text


def doc_to_target_special_formatted(doc):
    messages = doc.get("prompt", []) + doc.get("completion", [])
    assert messages[0]["role"] == "system", "System message is not the first message."
    assert messages[1]["role"] == "user", "User message is not the second message."
    assert messages[2]["role"] == "assistant", "Assistant message is not the third message."

    # replace in system message and assistant message's content tags <reasoning> </reasoning> with <think> and </think>
    messages[0]["content"] = messages[0]["content"].replace("<reasoning>", "<think>").replace("</reasoning>", "</think>")
    messages[2]["content"] = messages[2]["content"].replace("<reasoning>", "<think>").replace("</reasoning>", "</think>")

    message_text = TOKENIZER.apply_chat_template(
        conversation=messages,
        tokenize=False
    )

    return message_text


def doc_to_target_special_original(doc):
    instruction = doc.get("instruction", "")
    user_input = doc.get("input", "")
    output = doc.get("output", "")

    # Construct messages
    messages = []
    if instruction:
        messages.append({"role": "system", "content": instruction})
    if user_input:
        messages.append({"role": "user", "content": user_input})
    if output:
        messages.append({"role": "assistant", "content": output})

    assert len(messages) == 3, "There should be 3 messages."
    assert messages[0]["role"] == "system", "System message is not the first message."
    assert messages[1]["role"] == "user", "User message is not the second message."
    assert messages[2]["role"] == "assistant", "Assistant message is not the third message."

    message_text = TOKENIZER.apply_chat_template(
        conversation=messages,
        tokenize=False
    )

    return message_text


def doc_eval(pred, refs):
    try:
        bleu_results = bleu.compute(predictions=pred, references=refs)
    except Exception as e:
        print(f"Bleu error: {e}")
        bleu_results = {"bleu": np.nan}

    try:
        rouge_results = rouge.compute(predictions=pred, references=refs)
    except Exception as e:
        print(f"Rouge error: {e}")
        rouge_results = {"rouge1": np.nan, "rouge2": np.nan, "rougeL": np.nan}

    try:
        bert_scores = bertscore.compute(predictions=pred, references=refs, lang="en")[
            "f1"
        ]
    except Exception as e:
        print(f"Bert error: {e}")
        bert_scores = [np.nan]

    if bleu_results["bleu"] == 0:
        # Sometimes bleu is 0.0 and this breaks the stderr computation.
        bleu_results["bleu"] += 1e-5

    results = {
        "bleu": bleu_results["bleu"],
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bert_score": np.mean(bert_scores),
    }

    return results


def process_results_gen_formatted(doc, results):
    pred, refs = [results[0]], [doc_to_target_formatted(doc)]

    if len(refs[0]) < 1 or len(pred[0]) < 1:
        return {
            "bleu": np.nan,
            "rouge1": np.nan,
            "rouge2": np.nan,
            "rougeL": np.nan,
            "bert_score": np.nan,
        }

    results = doc_eval(pred, refs)

    return {
        "bleu": results["bleu"],
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"],
        "bert_score": results["bert_score"],
    }

def process_results_gen_original(doc, results):
    pred, refs = [results[0]], [doc_to_target_original(doc)]

    if len(refs[0]) < 1 or len(pred[0]) < 1:
        return {
            "bleu": np.nan,
            "rouge1": np.nan,
            "rouge2": np.nan,
            "rougeL": np.nan,
            "bert_score": np.nan,
        }

    results = doc_eval(pred, refs)

    return {
        "bleu": results["bleu"],
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"],
        "bert_score": results["bert_score"],
    }
