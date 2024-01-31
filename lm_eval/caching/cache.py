import os
import dill


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

OVERRIDE_PATH = os.getenv("LM_HARNESSS_CACHE_PATH")


PATH = OVERRIDE_PATH if OVERRIDE_PATH else f"{MODULE_DIR}/.cache"


def load_from_cache(file_name):
    try:
        path = f"{PATH}/{file_name}.pickle"

        with open(path, "rb") as file:
            cached_task_dict = dill.loads(file.read())
            return cached_task_dict

    except Exception as exception:
        print(f"{file_name} is not cached, generating...")
        pass


def save_to_cache(file_name, obj):
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    file_path = f"{PATH}/{file_name}.pickle"

    print(f"Saving {file_path} to cache...")
    with open(file_path, "wb") as file:
        file.write(dill.dumps(obj))
