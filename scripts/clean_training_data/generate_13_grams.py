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
import json
import pickle
import os
import sys
from pathlib import Path
import glob
import signal
from signal import SIGINT

from tqdm import tqdm

from lm_eval.decontamination.janitor import Janitor, word_ngrams
from lm_eval.decontamination.archiver import TextArchive, Reader

import logging
from tqdm_multiprocess.logger import setup_logger_tqdm

logger = logging.getLogger(__name__)

terminate = False
def handler(signal_received, frame):
    global terminate
    terminate = True

def get_pile(directory):
    reader = Reader()
    for file in glob.glob(os.path.join(directory, f"*.jsonl.zst*")):
        for document in reader.read(file):
            yield document

    def close_buckets(self):
        for bucket in self.buckets:
            bucket.commit()

def do_ngrams_in_buckets(n_value, working_directory, bucket_count):

    output_directory = os.path.join(working_directory, "output")
    os.makedirs(output_directory, exist_ok=True)

    logger.info(f"Generating {n_value}-grams and bucketing.")

    # Done file
    done_file = os.path.join(output_directory, f"ngram_buckets.done")
    if os.path.exists(done_file):
        logger.info("ngrams already generated and bucketed, skipping")
        return

    # Checkpoint
    checkpoint_file = os.path.join(working_directory, f"pile_offset.ckpt")
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
parser.add_argument("-sdir", "--save_directory", default="")
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