import os
import random
import string
from pathlib import Path

import numpy as np
import yaml
from pycocoevalcap.eval import Bleu, Cider, COCOEvalCap, Meteor, Rouge, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

# from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
# from lmms_eval.tasks._task_utils.video_loader import get_cache_dir

COCO_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]  # , "SPICE"]

from loguru import logger as eval_logger


def generate_submission_file(file_name, args, subpath="submissions"):
    path = os.path.join(args.output_path, subpath)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, file_name)
    return os.path.abspath(path)


def get_cache_dir(config, sub_dir="videos"):
    # HF_HOME = os.environ["HF_HOME"]
    cache_dir = config["dataset_kwargs"]["cache_dir"]
    # cache_dir = os.path.join(HF_HOME, cache_dir)
    cache_dir = os.path.join(cache_dir, sub_dir)
    # print(cache_dir)
    return cache_dir


def remove_nonascii(text):
    return "".join([i if ord(i) < 128 else " " for i in text])


def random_string(string_length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(string_length))


with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

cache_dir = get_cache_dir(config, "YouCookIIVideos")


def youcook2_doc_to_visual(doc):
    return [os.path.join(cache_dir, doc["video_path"])]


def youcook2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs and "prompt" in lmms_eval_specific_kwargs:
        return lmms_eval_specific_kwargs["prompt"]
    else:
        return "Provide a one-sentence caption for the provided video."


def youcook2_process_results(doc, result):
    pred = result[0] if result else ""
    video = doc["youtube_id"]
    timestamp = doc["segment"]

    data_dict = {"answer": remove_nonascii(doc["sentence"]), "pred": remove_nonascii(pred), "video": video, "timestamp": timestamp}

    return {f"{metric}": data_dict for metric in COCO_METRICS}


def youcook2_aggregate_results(results, metric, **kwargs):
    scorers = [(Bleu(4), "Bleu_1"), (Bleu(4), "Bleu_2"), (Bleu(4), "Bleu_3"), (Bleu(4), "Bleu_4"), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]  # , (Spice(), "SPICE")]
    scorers_dict = {s[1]: s[0] for s in scorers}

    gts = {}
    res = {}
    vid2capid = {}
    uid = 0
    cur_gts = {}
    cur_res = {}
    for result in results:
        if result["video"] not in vid2capid:
            vid2capid[result["video"]] = []
        vid2capid[result["video"]].append(uid)
        cur_gts[uid] = [{"caption": result["answer"]}]
        cur_res[uid] = [{"caption": result["pred"]}]
        uid += 1

    eval_logger.info("tokenization...")
    tokenizer = PTBTokenizer()
    tokenize_gts = tokenizer.tokenize(cur_gts)
    tokenize_res = tokenizer.tokenize(cur_res)

    eval_logger.info(f"Computing {metric} scores...")
    all_scores = []
    scorer = scorers_dict[metric]

    for vid_id, vid_list in vid2capid.items():
        res = {index: tokenize_res[index] for index in vid_list}
        gts = {index: tokenize_gts[index] for index in vid_list}

        if len(gts) == 0 or len(res) == 0:
            score = 0
        else:
            score, scores = scorer.compute_score(gts, res)
        all_scores.append(score)
    return np.mean(all_scores) * 100


def youcook2_bleu4(results, **kwargs):
    return youcook2_aggregate_results(results, "Bleu_4", **kwargs)


def youcook2_bleu3(results, **kwargs):
    return youcook2_aggregate_results(results, "Bleu_3", **kwargs)


def youcook2_bleu2(results, **kwargs):
    return youcook2_aggregate_results(results, "Bleu_2", **kwargs)


def youcook2_bleu1(results, **kwargs):
    return youcook2_aggregate_results(results, "Bleu_1", **kwargs)


def youcook2_meteor(results, **kwargs):
    return youcook2_aggregate_results(results, "METEOR", **kwargs)


def youcook2_rougel(results, **kwargs):
    return youcook2_aggregate_results(results, "ROUGE_L", **kwargs)


def youcook2_cider(results, **kwargs):
    return youcook2_aggregate_results(results, "CIDEr", **kwargs)


def youcook2_spice(results, args):
    return youcook2_aggregate_results(results, "SPICE", args)