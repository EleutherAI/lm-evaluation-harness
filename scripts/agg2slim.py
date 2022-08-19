import glob
import json
import os
import logging


logger = logging.getLogger(__name__)


def agg2slim(data):
    """Maps results data to a simpler dictionary.

    `data` is expected to have a `results` and `config` fields.
    The results should be a list of dictionaries. This function
    filters out some of that information. `config` should be a dictionary.
    """
    REMOVE_KEYS = set(
        [
            "fixed_answer_choice_list",
            "prompt_id",
            "prompt_jinja",
            "prompt_original_task",
            "comment",
        ]
    )
    results = data["results"]
    config = data["config"]
    if isinstance(config, list):
        logger.warning("Warning! This is an old agg file with a buggy config.")
        # If information needs to be recovered, we can recover it from the filename.
        config = {}
    slim = {
        "results": [
            {k: v for k, v in r.items() if k not in REMOVE_KEYS} for r in results
        ],
        "config": config,
    }
    return slim


def main():
    """Reformats each `agg` json file into a smaller slim version.

    python -m scripts/agg2slim.py
    """
    os.makedirs("./outputs", exist_ok=True)
    for agg_json_filename in glob.glob("./outputs/agg*.json"):
        with open(agg_json_filename, "r") as jf:
            data = json.load(jf)
        slim_json_filename = agg_json_filename.replace("agg", "slim")
        if os.path.exists(slim_json_filename):
            logger.info("Skipping file as it already exists.")
        slim = agg2slim(data)
        with open(slim_json_filename, "w") as jf:
            json.dump(slim, jf, indent=2)


if __name__ == "__main__":
    main()
