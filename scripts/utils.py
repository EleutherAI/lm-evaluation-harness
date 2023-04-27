from pathlib import Path
import shutil


def rmrf_then_mkdir(path: Path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        ...

    path.mkdir(exist_ok=True, parents=True)
