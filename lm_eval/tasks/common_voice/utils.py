from typing import Any


INSTRUCTION = (
    "Listen to the audio <audio> and output what a human has said on it. Answer:"
)


def doc_to_text(doc: dict[str, Any]) -> str:
    return INSTRUCTION


def doc_to_audio(doc: dict[str, Any]) -> list[dict]:
    audio = {
        "array": doc["audio"]["array"],
        "sampling_rate": doc["audio"]["sampling_rate"],
    }

    audios = [audio]
    return audios
