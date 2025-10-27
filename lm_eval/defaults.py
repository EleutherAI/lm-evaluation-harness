import os


def to_bool(value: str | bool) -> bool:
    return str(value).lower() in ("yes", "true", "t", "1")


ALL_OUTPUT_TYPES = [
    "loglikelihood",
    "multiple_choice",
    "loglikelihood_rolling",
    "generate_until",
]
DEFAULT_MAX_LENGTH = int(os.environ.get("LMEVAL_DEFAULT_MAX_LENGTH", 2048))
DEFAULT_GEN_MAX_LENGTH = int(os.environ.get("LMEVAL_DEFAULT_GEN_MAX_LENGTH", 256))
DATA_VALIDATION = to_bool(os.environ.get("LMEVAL_DATA_VALIDATION", False))
DISABLE_MULTIPROC = to_bool(os.environ.get("LMEVAL_DISABLE_MULTIPROC", False))
HASH_MULTIMODAL = to_bool(os.environ.get("LMEVAL_HASHMM", True))
