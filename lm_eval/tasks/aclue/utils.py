"""Load the original ACLUE CSVs without executing a remote dataset script."""

from pathlib import Path

import datasets


DATA_URL = (
    "https://huggingface.co/datasets/tyouisen/aclue/resolve/"
    "f89544164a29a9eec66490bf6d0cd571ceab4358/aclue_v1_0_0.zip"
)


def load_dataset(subject, cache_dir=None, **kwargs):
    """Preserve the original loader's splits, string fields, and file order."""
    manager = datasets.DownloadManager(
        download_config=datasets.DownloadConfig(cache_dir=cache_dir)
    )
    data_dir = Path(manager.download_and_extract(DATA_URL))
    return datasets.load_dataset(
        "csv",
        name=subject,
        data_files={
            split: str(data_dir / split / f"{subject}.csv") for split in ("test", "dev")
        },
        features=datasets.Features(
            {
                column: datasets.Value("string")
                for column in ("Question", "A", "B", "C", "D", "Answer")
            }
        ),
        encoding="utf-8",
        cache_dir=cache_dir,
    )
