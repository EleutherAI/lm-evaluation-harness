from lm_eval.decontamination.archiver import Reader
import os
import json
from functools import reduce
import glob
import tqdm

from tqdm_multiprocess import TqdmMultiProcessPool


def get_file_stats(file_path, tqdm_func, global_tqdm):
    reader = Reader()
    total_documents = 0
    total_size = 0
    update_frequency = 10000
    current_file_position = 0

    with tqdm_func(
        total=os.path.getsize(file_path), dynamic_ncols=True, unit="byte", unit_scale=1
    ) as progress:
        for document in reader.read(file_path, get_meta=True):
            total_size += len(document)
            total_documents += 1

            if total_documents % update_frequency == 0:
                new_file_pos = reader.fh.tell()
                bytes_read = new_file_pos - current_file_position
                current_file_position = new_file_pos
                progress.update(bytes_read)
                global_tqdm.update(bytes_read)

    return (total_documents, total_size)


def get_files():
    directory = "pile"
    files = list(sorted(glob.glob(os.path.join(directory, "*.jsonl.zst*"))))
    print(files)
    return files


def get_stats():
    files = get_files()
    total_size_bytes = sum(map(lambda x: os.path.getsize(x), files))

    pool = TqdmMultiProcessPool(4)
    global_tqdm = tqdm.tqdm(
        total=total_size_bytes, dynamic_ncols=True, unit="byte", unit_scale=1
    )

    # Generate minhashes with pool
    tasks = [(get_file_stats, (file,)) for file in files]

    def on_done(_):
        return None

    def on_error(_):
        return None

    results = pool.map(global_tqdm, tasks, on_error, on_done)

    total_documents, total_size = reduce(
        lambda x, y: (x[0] + y[0], x[1] + y[1]), results
    )

    start_offsets = []
    current_offset = 0
    for file_document_count, _ in results:
        start_offsets.append(current_offset)
        current_offset += file_document_count

    return (total_documents, total_size, start_offsets)


if __name__ == "__main__":
    version = 1.01
    print(f"Running version {version}")

    stats_file_path = "pile_statistics.json"
    if os.path.exists(stats_file_path):
        stats = json.load(open(stats_file_path, "r"))
    else:
        document_count, total_document_size_chars, start_offsets = get_stats()
        stats = {
            "Data": "Pile statistics",
            "Document Count": document_count,
            "Total Pile Characters": total_document_size_chars,
            "File Start Offsets": start_offsets,
        }
        json.dump(stats, open(stats_file_path, "w"), indent=4)

    print(f"document_count: {stats['Document Count']}")
    print(f"total_chars: {stats['Total Pile Characters']}")
    print(f"start_offsets: {stats['File Start Offsets']}")
