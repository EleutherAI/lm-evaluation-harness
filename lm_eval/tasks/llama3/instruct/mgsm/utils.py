import re
import string
from functools import partial
from typing import TYPE_CHECKING, Dict, List


if TYPE_CHECKING:
    import datasets

from lm_eval.api.metrics import exact_match_fn


TRANSLATE_TABLE = str.maketrans(
    "", "", string.punctuation.replace(".", "")
)  # decimals are handled by the number_variations function
# extracted from https://huggingface.co/datasets/meta-llama/Llama-3.2-3B-Instruct-evals/viewer/Llama-3.2-3B-Instruct-evals__mgsm__details
PROMPTS = [
    {
        "rep": 'Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".',
        "subtask_name": "en",
    },
    {
        "rep": 'Решите эту математическую задачу. Объясните шаги рассуждения перед тем, как дать окончательный ответ в последней строке сам по себе в формате "Ответ:". Не добавляйте ничего, кроме целочисленного ответа после "Ответ:".',
        "subtask_name": "ru",
    },
    {
        "rep": 'Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".',
        "subtask_name": "sw",
    },
    {
        "rep": 'Résolvez ce problème de mathématiques. Donnez les étapes de raisonnement avant de fournir la réponse finale sur la dernière ligne elle-même dans le format de "Réponse:". N\'ajoutez rien d\'autre que la réponse entière après "Réponse:".',
        "subtask_name": "fr",
    },
    {
        "rep": "ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి. చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ణాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.",
        "subtask_name": "te",
    },
    {
        "rep": 'แก้ปัญหาคณิตศาสตร์นี้ ให้ให้ขั้นตอนการใช้เหตุผลก่อนที่จะให้คำตอบสุดท้ายในบรรทัดสุดท้ายโดยอยู่ในรูปแบบ "คำตอบ:" ไม่ควรเพิ่มอะไรนอกจากคำตอบที่เป็นจำนวนเต็มหลังจาก "คำตอบ:',
        "subtask_name": "th",
    },
    {
        "rep": 'の数学の問題を解いてください。最終的な答えを出す前に、解答の推論過程を記述してください。そして最後の行には "答え:" の形式で答えを記述し、その後には整数の答え以外何も追加しないでください。',
        "subtask_name": "ja",
    },
    {
        "rep": 'Löse dieses Mathematikproblem. Gib die Schritte zur Begründung an, bevor du die endgültige Antwort in der letzten Zeile alleine im Format "Antwort:" gibst. Füge nichts anderes als die ganzzahlige Antwort nach "Antwort:" hinzu.',
        "subtask_name": "de",
    },
    {
        "rep": 'এই গণিতের সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিসম্পন্ন পদক্ষেপ প্রদান করুন। চূড়ান্ত উত্তরটি একক সংখ্যা হিসাবে "উত্তর:" এর পরে শেষ লাইনে দিন। "উত্তর:" এর পরে অন্য কিছু যুক্ত করবেন না।.',
        "subtask_name": "bn",
    },
    {
        "rep": '解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。',
        "subtask_name": "zh",
    },
    {
        "rep": 'Resuelve este problema matemático. Proporciona los pasos de razonamiento antes de dar la respuesta final en la última línea por sí misma en el formato de "Respuesta:". No añadas nada más que la respuesta entera después de "Respuesta:".',
        "subtask_name": "es",
    },
]


def number_variations(n: int) -> List[str]:
    formats = []
    # Generate each pattern twice
    for _ in range(2):
        # Basic string representation
        formats.append(str(n))
        formats.append(f"{n}.")

        # With one decimal place
        formats.append(f"{n}.0")
        formats.append(f"{n}.0.")

        # With two decimal places
        formats.append(f"{n}.00")
        formats.append(f"{n}.00.")

    return formats


def process_docs(lang: str, df: "datasets.Dataset") -> "datasets.Dataset":
    def map_(doc: dict):
        suffix = [x for x in PROMPTS if x["subtask_name"] == lang][0]["rep"]

        doc["question"] = (
            suffix
            + "\n\n"
            + re.split("[:|：]", doc["question"], maxsplit=1)[-1].strip()
        )
        doc["answers"] = number_variations(doc["answer_number"])
        return doc

    return df.map(map_)


process_docs_bn = partial(process_docs, "bn")
process_docs_de = partial(process_docs, "de")
process_docs_en = partial(process_docs, "en")
process_docs_es = partial(process_docs, "es")
process_docs_fr = partial(process_docs, "fr")
process_docs_ja = partial(process_docs, "ja")
process_docs_ru = partial(process_docs, "ru")
process_docs_sw = partial(process_docs, "sw")
process_docs_te = partial(process_docs, "te")
process_docs_th = partial(process_docs, "th")
process_docs_zh = partial(process_docs, "zh")


def process_results(doc: dict, prediction: List[str]) -> Dict[str, int]:
    gold: List = doc["answers"]
    return {
        "exact_match": int(
            exact_match_fn(
                predictions=[x.strip().translate(TRANSLATE_TABLE) for x in prediction]
                * len(gold),
                references=gold,
                ignore_case=True,
                ignore_punctuation=False,
            )["exact_match"]
            > 0
        )
    }
