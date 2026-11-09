import argparse
import glob
import logging
import os
import shutil
import subprocess

from tqdm import tqdm
from tqdm_multiprocess import TqdmMultiProcessPool
from tqdm_multiprocess.logger import setup_logger_tqdm


logger = logging.getLogger(__name__)


def process_task(
    working_directory, output_directory, bucket_file_path, tqdm_func, global_tqdm
):
    command = f"zstd {bucket_file_path}"
    logger.info(command)
    subprocess.call(command, shell=True)

    compressed_file = bucket_file_path + ".zst"
    if output_directory:
        shutil.move(compressed_file, output_directory)

    os.remove(bucket_file_path)
    global_tqdm.update()


def compress_and_move(working_directory, output_directory, process_count):
    os.makedirs(output_directory, exist_ok=True)
    original_info_file_path = os.path.join(working_directory, "info.json")
    assert os.path.exists(original_info_file_path)

    tasks = []
    bucket_file_paths = glob.glob(
        os.path.join(working_directory, "output", "*.bkt.txt.sorted")
    )
    for bucket_file_path in bucket_file_paths:
        task = (process_task, (working_directory, output_directory, bucket_file_path))
        tasks.append(task)

    pool = TqdmMultiProcessPool(process_count)

    def on_done(_):
        return None

    def on_error(_):
        return None

    global_progress = tqdm(
        total=len(bucket_file_paths), dynamic_ncols=True, unit="file"
    )
    _ = pool.map(global_progress, tasks, on_error, on_done)

    shutil.copy(original_info_file_path, os.path.join(output_directory, "info.json"))


parser = argparse.ArgumentParser(description="sort 13gram buckets")
parser.add_argument("-dir", "--working_directory", required=True)
parser.add_argument("-output", "--output_directory", required=True)
parser.add_argument("-procs", "--process_count", type=int, default=8)

if __name__ == "__main__":
    version = 1.00
    print(f"Running version {version}")

    logfile_path = "compress_and_package.log"
    setup_logger_tqdm(logfile_path)

    args = parser.parse_args()
    compress_and_move(args.working_directory, args.output_directory, args.process_count)
