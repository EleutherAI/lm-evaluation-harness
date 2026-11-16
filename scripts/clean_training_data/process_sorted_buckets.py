"""
Processes each sorted bucket, creating a new file listing all ngrams that matched more then 10
unique documents with their unique document counts. Uses multiprocessing and very little memory
as we stream from presorted buckets. Will use a lot of disk though.

Arguments
---------
--working_directory (-dir)
    Directory containing the sorted buckets, processed files will be deposited here. Default: current directory
--move_dir (-move)
    Directory to move processed 13grams too. Default: Do nothing
--process_count (-procs)
    Number of processes to use. Default: 4
"""

import argparse
import glob
import logging
import os
import re
import shutil
from pathlib import Path

from tqdm import tqdm
from tqdm_multiprocess import TqdmMultiProcessPool
from tqdm_multiprocess.logger import setup_logger_tqdm

from scripts.clean_training_data.archiver import TextArchive, TextReader


logger = logging.getLogger(__name__)


# Multiprocessed
def process_bucket(
    bucket_file_path, processed_directory, move_dir, tqdm_func, global_tqdm
):
    bucket_id = re.sub("\D", "", os.path.basename(bucket_file_path))  # noqa: W605
    done_file = os.path.join(
        processed_directory, f"ngram_bucket_processing_{bucket_id}.done"
    )
    if os.path.exists(done_file):
        logger.info(f"bucket {bucket_id} already processed, skipping")
        return

    # For managing tqdm
    file_size = os.path.getsize(bucket_file_path)
    bucket_progress = tqdm_func(
        total=file_size, dynamic_ncols=True, unit="byte", unit_scale=1
    )
    current_file_position = 0
    update_frequency = 100 * 1000000  # 100mb
    update_counter = 0

    # Iterate through and output ngrams which occur in more then 10 documents
    bucket = TextReader(bucket_file_path)

    output_file_path = bucket_file_path + ".processed"
    output_archive = TextArchive(output_file_path, mode="wb")

    current_ngram = ""
    current_ngram_document_ids = set()
    for line in bucket.read():
        [ngram, document_id] = line.rsplit(" ", 1)

        # Write ngram if more then 10 unique document occurrences
        if ngram != current_ngram:
            if len(current_ngram_document_ids) > 10:
                output_archive.add_data(
                    f"{current_ngram} {len(current_ngram_document_ids)}"
                )
            current_ngram = ngram
            current_ngram_document_ids = set()

        current_ngram_document_ids.add(document_id)

        # Update tqdm
        update_counter += bucket.fh.tell() - current_file_position
        current_file_position = bucket.fh.tell()
        if update_counter > update_frequency:
            bucket_progress.update(update_counter)
            update_counter = 0

    # Remainder
    if len(current_ngram_document_ids) > 10:
        output_archive.add_data(f"{current_ngram} {len(current_ngram_document_ids)}")

    output_archive.commit()
    Path(done_file).touch()

    if move_dir:
        shutil.move(output_file_path, move_dir)

    global_tqdm.update()


def process_sorted_buckets(working_directory, move_dir, process_count):
    bucket_file_paths = glob.glob(os.path.join(working_directory, "*.bkt.txt.sorted"))
    processed_directory = os.path.join(working_directory, "processed")
    os.makedirs(processed_directory, exist_ok=True)

    pool = TqdmMultiProcessPool(process_count)
    tasks = [
        (process_bucket, (bucket_file, processed_directory, move_dir))
        for bucket_file in bucket_file_paths
    ]

    global_tqdm = tqdm(total=len(bucket_file_paths), dynamic_ncols=True, unit="bucket")

    def on_done(_):
        return None

    def on_error(_):
        return None

    _ = pool.map(global_tqdm, tasks, on_error, on_done)


parser = argparse.ArgumentParser(description="Process 13 grams from sorted buckets.")
parser.add_argument("-dir", "--working_directory", default="")
parser.add_argument("-move", "--move_dir", default="")
parser.add_argument("-procs", "--process_count", type=int, default=4)

if __name__ == "__main__":
    logfile_path = "process13grams.log"
    setup_logger_tqdm(logfile_path)

    args = parser.parse_args()
    process_sorted_buckets(args.working_directory, args.move_dir, args.process_count)
