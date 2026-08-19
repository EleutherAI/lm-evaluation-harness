import hashlib
import logging
import os
import re

import dill


eval_logger = logging.getLogger(__name__)


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

OVERRIDE_PATH = os.getenv("LM_HARNESS_CACHE_PATH")


PATH = OVERRIDE_PATH or f"{MODULE_DIR}/.cache"

# This should be sufficient for uniqueness
HASH_INPUT = "EleutherAI-lm-evaluation-harness"

HASH_PREFIX = hashlib.sha256(HASH_INPUT.encode("utf-8")).hexdigest()

FILE_SUFFIX = f".{HASH_PREFIX}.pickle"

# Keep cache file basenames comfortably below common 255-byte filesystem limits.
# Some request cache keys include model paths or endpoint identifiers, which can be
# very long. Preserve a readable prefix and append a hash of the full key so cache
# entries remain stable and unique without overflowing the filesystem limit.
MAX_CACHE_FILENAME_BYTES = 240
CACHE_KEY_HASH_LEN = 16


def _cache_file_path(file_name: str) -> str:
    safe_file_name = re.sub(r"[/\\\\]+", "_", file_name)
    basename = f"{safe_file_name}{FILE_SUFFIX}"

    if len(basename.encode("utf-8")) <= MAX_CACHE_FILENAME_BYTES:
        return os.path.join(PATH, basename)

    file_name_hash = hashlib.sha256(file_name.encode("utf-8")).hexdigest()[
        :CACHE_KEY_HASH_LEN
    ]
    hashed_suffix = f"-{file_name_hash}{FILE_SUFFIX}"
    max_prefix_bytes = MAX_CACHE_FILENAME_BYTES - len(hashed_suffix.encode("utf-8"))

    prefix = safe_file_name.encode("utf-8")[:max_prefix_bytes]
    prefix = prefix.decode("utf-8", errors="ignore").rstrip("-_.") or "cache"

    return os.path.join(PATH, f"{prefix}{hashed_suffix}")


def load_from_cache(file_name: str, cache: bool = False):
    if not cache:
        return
    try:
        path = _cache_file_path(file_name)

        with open(path, "rb") as file:
            cached_task_dict = dill.loads(file.read())  # noqa: S301
            return cached_task_dict

    except Exception:  # noqa: BLE001
        eval_logger.debug("%s is not cached, generating...", file_name)


def save_to_cache(file_name, obj):
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    file_path = _cache_file_path(file_name)

    eval_logger.debug("Saving %s to cache...", file_path)
    with open(file_path, "wb") as file:
        file.write(dill.dumps(obj))


# NOTE the "key" param is to allow for flexibility
def delete_cache(key: str = ""):
    files = os.listdir(PATH)

    for file in files:
        if file.startswith(key) and file.endswith(FILE_SUFFIX):
            file_path = f"{PATH}/{file}"
            os.unlink(file_path)
