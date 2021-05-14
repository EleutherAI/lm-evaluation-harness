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
import pickle
import os
from pathlib import Path
import glob
import signal
from signal import SIGINT

from tqdm import tqdm

from scripts.clean_training_data.janitor import Janitor, word_ngrams
from scripts.clean_training_data.archiver import TextArchive, Reader

import logging
from tqdm_multiprocess.logger import setup_logger_tqdm
logger = logging.getLogger(__name__)

pile_document_count = 210607728

terminate = False
def handler(signal_received, frame):
    global terminate
    terminate = True

def get_pile(directory):
    reader = Reader()
    for file in glob.glob(os.path.join(directory, f"*.jsonl.zst*")):
        for document in reader.read(file):
            yield document

def close_buckets(buckets):
    for bucket in buckets:
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
    checkpoint_file = os.path.join(output_directory, f"ngram_buckets.ckpt")
    if os.path.exists(checkpoint_file):
        start_id = pickle.load(open(checkpoint_file,"rb"))
    else:
        start_id = 0

    logger.info(f"Starting at pile document index {start_id}")
    bucket_files = [os.path.join(output_directory, f"ngrams_{i}.bkt.txt") for i in range(bucket_count)]
    buckets = list(map(TextArchive, bucket_files))

    janitor = Janitor()
    current_id = 0
    batch_size = 1000
    batch_counter = 0
    with tqdm(total=pile_document_count, dynamic_ncols=True, unit="docs") as progress:
        for document in get_pile(working_directory):
            if current_id < start_id:
                if terminate:
                    close_buckets(buckets)
                    return

                current_id += 1
                progress.update()
                continue

            # Save checkpoint every "batch_size", only allow terminate after checkpoint
            if batch_counter == batch_size:
                progress.update(batch_size)
                batch_counter = 0
                pickle.dump(current_id, open(checkpoint_file,"wb"))
                if terminate:
                    close_buckets(buckets)
                    return

            ngrams = word_ngrams(janitor.normalize_string(document), n_value)
            for ngram in ngrams:
                bucket = hash(ngram) % len(buckets)
                buckets[bucket].add_data(f"{ngram} {current_id}")

            batch_counter += 1
            current_id += 1
    
    close_buckets(buckets)
    Path(done_file).touch()


parser = argparse.ArgumentParser(description='Generate 13 grams from Pile.')
parser.add_argument("-dir", "--working_directory", default="")
parser.add_argument("-n", "--n_value", type=int, default=13)
parser.add_argument("-buckets", "--bucket_count", type=int, default=500)

if __name__ == '__main__':

    # Handle sigint (ctrl-c) cleanly
    previous_signal_int = signal.signal(SIGINT, handler)

    logfile_path = "ngrams.log"
    setup_logger_tqdm(logfile_path)

    args = parser.parse_args()
    do_ngrams_in_buckets(args.n_value, args.working_directory, args.bucket_count)