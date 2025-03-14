import io
from typing import Any, Dict, List


INSTRUCTION = (
    "Listen to the audio <audio> and output what a human has said on it. Answer:"
)


def doc_to_text(doc: Dict[str, Any]) -> str:
    return INSTRUCTION


def doc_to_audio(doc: Dict[str, Any]) -> List[dict]:
    audio = {
        "array": doc["audio"]["array"],
        "sampling_rate": doc["audio"]["sampling_rate"],
    }

    audios = [audio]
    return audios
