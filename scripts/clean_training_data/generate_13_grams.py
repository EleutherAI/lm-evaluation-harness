"""
Outputs all 13-grams found in The Pile.

Loops through all documents and uses the logic found in janitor.py to extract 13-grams.
We bucket each 13-gram by hash into separate file buckets to allow easy parallel processing in the
next stage. We also include the current pile document_id with each ngram instance to allow the
filtering to exclude 13-grams that match more then 10 unique documents (done further down the pipeline).

We didn't use lm_dataformat to output as it increases time 4x (slow jsonify) and makes
resuming hard (and we had the storage).

Arguments
---------
--working_directory (-dir)
    Directory containing the pile distribution. An "output" subdirectory will be created underneath
    to store the bucketed 13-grams, checkpoint and done files. Default: current directory
--n_value (-n)
    n value in n-gram, added for later use if ever needed. Default: 13
--bucket_count (-buckets)
    Number of file buckets to use when generating 13grams. Default: 500
"""

import argparse
import glob
import json
import logging
import os
import pickle
import signal
import sys
from pathlib import Path
from signal import SIGINT

from tqdm import tqdm
from tqdm_multiprocess.logger import setup_logger_tqdm

from lm_eval.decontamination.archiver import Reader, TextArchive
from lm_eval.decontamination.janitor import Janitor, word_ngrams


logger = logging.getLogger(__name__)

terminate = False


def handler(signal_received, frame):
    global terminate
    terminate = True


def yield_pile(start_offsets=None, checkpoint_offset=None):
    directory = "pile"

    if not os.path.exists(directory):
        print(
            "We expect the pile archives to be in the 'pile' directory, but this was not found."
        )
        raise Exception("Pile directory not found.")

    files = list(sorted(glob.glob(os.path.join(directory, "*.jsonl.zst*"))))

    pile_global_offset = 0
    start_file = 0
    if checkpoint_offset:
        for file_i, start_offset in enumerate(start_offsets):
            if start_offset > checkpoint_offset:
                break

            start_file = file_i
            pile_global_offset = start_offset

    for file_i, file in enumerate(files):
        if file_i < start_file:
            logger.info(f"Skipping file {file}")
            continue
        logger.info(f"Reading from pile file: {file}")
        reader = Reader()
        for document in reader.read(file):
            yield (pile_global_offset, document)
            pile_global_offset += 1


# Hash buckets > disk backed files. Supports file position checkpointing and resuming
# Allows you to write continuously and checkpoint intermittently. If a failure occurs
# the buckets are simply truncated at your last checkpoint.
class Buckets:
    def __init__(self, directory, num_buckets):
        self.bucket_files = [
            os.path.join(directory, f"ngrams_{i}.bkt.txt") for i in range(num_buckets)
        ]
        self.buckets = list(map(TextArchive, self.bucket_files))
        self.checkpoint_file = os.path.join(directory, "bucket_offsets.ckpt")

        if os.path.exists(self.checkpoint_file):
            self.bucket_offsets = pickle.load(open(self.checkpoint_file, "rb"))
        else:
            self.bucket_offsets = [0 for i in range(len(self.buckets))]

        for i, offset in enumerate(self.bucket_offsets):
            bucket = self.buckets[i]
            bucket.fh.seek(offset)
            bucket.fh.truncate()

    def add_data(self, key, value):
        i = hash(key) % len(self.buckets)
        bucket = self.buckets[i]
        bucket.add_data(value)

    def save_checkpoint(self):
        for bucket in self.buckets:
            bucket.fh.flush()

        bucket_offsets = [bucket.fh.tell() for bucket in self.buckets]
        pickle.dump(bucket_offsets, open(self.checkpoint_file, "wb"))

    def close_buckets(self):
        for bucket in self.buckets:
            bucket.commit()


def do_ngrams_in_buckets(n_value, working_directory, bucket_count):
    pile_statistics = json.load(open("pile_statistics.json", "r", encoding="utf-8"))
    pile_document_count = pile_statistics["Document Count"]
    start_offsets = pile_statistics["File Start Offsets"]

    output_directory = os.path.join(working_directory, "output")
    os.makedirs(output_directory, exist_ok=True)

    logger.info(f"Generating {n_value}-grams and bucketing.")

    # Done file
    done_file = os.path.join(output_directory, "ngram_buckets.done")
    if os.path.exists(done_file):
        logger.info("ngrams already generated and bucketed, skipping")
        return

    # Checkpoint
    checkpoint_file = os.path.join(working_directory, "pile_offset.ckpt")
    if os.path.exists(checkpoint_file):
        checkpoint_offset = pickle.load(open(checkpoint_file, "rb"))
        iterate = True
    else:
        checkpoint_offset = 0
        iterate = False

    logger.info(f"Starting at pile document index {checkpoint_offset}")
    buckets = Buckets(output_directory, bucket_count)

    janitor = Janitor()
    batch_size = 1000
    batch_counter = 0

    with tqdm(total=checkpoint_offset, dynamic_ncols=True, unit="docs") as progress:
        for offset, document in yield_pile(start_offsets, checkpoint_offset):
            if iterate:
                logger.info(f"Iterating to offset {checkpoint_offset} from {offset}")
                progress.update(offset)
                iterate = False

            if offset < checkpoint_offset:
                progress.update()

                if terminate:
                    return
                continue

            if offset == checkpoint_offset:
                progress.reset(total=pile_document_count)
                progress.update(checkpoint_offset)

            # Save checkpoint every "batch_size", only allow terminate after checkpoint
            if batch_counter == batch_size:
                progress.update(batch_size)
                batch_counter = 0
                buckets.save_checkpoint()
                pickle.dump(offset, open(checkpoint_file, "wb"))
                if terminate:
                    buckets.close_buckets()
                    return

            ngrams = word_ngrams(janitor.normalize_string(document), n_value)
            for ngram in ngrams:
                buckets.add_data(ngram, f"{ngram} {offset}")

            batch_counter += 1

    buckets.close_buckets()
    Path(done_file).touch()


parser = argparse.ArgumentParser(description="Generate 13 grams from Pile.")
parser.add_argument("-dir", "--working_directory", default="")
parser.add_argument("-n", "--n_value", type=int, default=13)
parser.add_argument("-buckets", "--bucket_count", type=int, default=500)

if __name__ == "__main__":
    version = 1.00
    print(f"Running version {version}")

    if "PYTHONHASHSEED" not in os.environ or os.environ["PYTHONHASHSEED"] != "0":
        print("Please run 'export PYTHONHASHSEED=0' before running generate.")
        sys.exit()

    # Handle sigint (ctrl-c) cleanly
    previous_signal_int = signal.signal(SIGINT, handler)

    logfile_path = "ngrams.log"
    setup_logger_tqdm(logfile_path)

    args = parser.parse_args()
    do_ngrams_in_buckets(args.n_value, args.working_directory, args.bucket_count)

    info_dict = {"title": "dataset ngrams", "ngram_size": 13}
    info_dict_path = os.path.join(args.working_directory, "info.json")
    json.dump(info_dict, open(info_dict_path, "w", encoding="utf-8"))
