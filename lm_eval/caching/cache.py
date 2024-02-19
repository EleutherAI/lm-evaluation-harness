import os
import dill
import hashlib
import random


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

OVERRIDE_PATH = os.getenv("LM_HARNESS_CACHE_PATH")


PATH = OVERRIDE_PATH if OVERRIDE_PATH else f"{MODULE_DIR}/.cache"

# this is to effectively eliminate the possibility of a cache collision
random.seed("EleutherAI")

# we probably don't need a salt for this, but more chaos = lower liklihood of collision
SALT = random.random()

HASH_INPUT = f"{SALT}-EleutherAI-lm-evaluation-harness"

HASH_PREFIX = hashlib.sha256(HASH_INPUT.encode("utf-8")).hexdigest()

# we just append the hash to the file so it's still easy to see what saved what
# e
FILE_SUFFIX = f".{HASH_PREFIX}.pickle"


def load_from_cache(file_name):
    try:
        path = f"{PATH}/{file_name}{FILE_SUFFIX}"

        with open(path, "rb") as file:
            cached_task_dict = dill.loads(file.read())
            return cached_task_dict

    except Exception as exception:
        print(f"{file_name} is not cached, generating...")
        pass


def save_to_cache(file_name, obj):
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    file_path = f"{PATH}/{file_name}{FILE_SUFFIX}"

    print(f"Saving {file_path} to cache...")
    with open(file_path, "wb") as file:
        file.write(dill.dumps(obj))


# NOTE the "key" param is to allow for flexibility
def delete_cache(key: str = ""):
    files = os.listdir(PATH)

    for file in files:
        if file.startswith(key) and file.endswith(FILE_SUFFIX):
            file_path = f"{PATH}/{file}"
            os.unlink(file_path)
