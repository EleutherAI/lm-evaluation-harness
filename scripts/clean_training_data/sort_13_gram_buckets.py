"""
Iteratively runs gnu sort on each bucket, uses up to 8 cores.

Arguments
---------
--working_directory (-dir)
    Directory containing the bucketed 13-grams. Sorted buckets will be deposited in the same
    directory and the unsorted buckets are removed after.
"""

import glob
import argparse
import os
import signal
from signal import SIGINT
import subprocess

from tqdm import tqdm

import logging
from tqdm_multiprocess.logger import setup_logger_tqdm

logger = logging.getLogger(__name__)

terminate = False


def handler(signal_received, frame):
    global terminate
    terminate = True


def sort_13_gram_buckets(working_directory):
    bucket_file_paths = glob.glob(os.path.join(working_directory, f"*.bkt.txt"))

    for bucket_file_path in tqdm(bucket_file_paths, dynamic_ncols=True):
        sorted_file_path = bucket_file_path + ".sorted"
        command = f"sort {bucket_file_path} > {sorted_file_path}"
        logger.info(command)
        subprocess.call(command, shell=True)

        if terminate:
            return

        os.remove(bucket_file_path)


parser = argparse.ArgumentParser(description="sort 13gram buckets")
parser.add_argument("-dir", "--working_directory", default="")

if __name__ == "__main__":

    version = 1.00
    print(f"Running version {version}")

    # Handle sigint (ctrl-c) cleanly
    previous_signal_int = signal.signal(SIGINT, handler)

    logfile_path = "sort13grambuckets.log"
    setup_logger_tqdm(logfile_path)

    args = parser.parse_args()
    sort_13_gram_buckets(args.working_directory)
