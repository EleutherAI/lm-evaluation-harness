import gzip
import json
import logging
import os
import re
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional

import datasets
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


LOGGER = logging.getLogger(__name__)

TEXT_FIELD_CANDIDATES: tuple[str, ...] = (
    "text",
    "body",
    "content",
    "article",
    "document",
    "raw_text",
    "code",
    "message",
    "description",
    "story",
)

LIST_FIELD_CANDIDATES: tuple[str, ...] = (
    "paragraphs",
    "sentences",
    "lines",
    "messages",
)

FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>.+)_(?P<start>\d{8})to(?P<end>\d{8})(?P<suffix>(?:\.[^.]+)*)$"
)

AUTO_DOWNLOAD_ENABLED = os.getenv(
    "UNCHEATABLE_EVAL_DISABLE_DOWNLOAD", ""
).lower() not in {"1", "true", "yes"}

DEFAULT_CACHE_DIR = Path(
    os.getenv(
        "UNCHEATABLE_EVAL_CACHE_DIR",
        Path.home() / ".cache" / "uncheatable_eval" / "latest",
    )
)

GITHUB_REQUEST_TIMEOUT = int(os.getenv("UNCHEATABLE_EVAL_REQUEST_TIMEOUT", "120"))


def _resolve_data_root(data_root: Optional[str] = None) -> Path:
    """Return the directory containing Uncheatable Eval dumps."""

    if data_root:
        path = Path(data_root).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    env_root = os.getenv("UNCHEATABLE_EVAL_DATA_ROOT") or os.getenv(
        "UNCHEATABLE_EVAL_ROOT"
    )
    if env_root:
        path = Path(env_root).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    candidates: List[Path] = []

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidates.extend(
            [
                parent / "uncheatable_eval" / "data",
                parent / "uncheatable-eval" / "data",
                parent / "raw" / "uncheatable-eval" / "latest",
                parent / "raw" / "uncheatable_eval" / "latest",
                parent / "local_store" / "raw" / "uncheatable-eval" / "latest",
                parent / "local_store" / "raw" / "uncheatable_eval" / "latest",
            ]
        )

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "uncheatable_eval" / "data",
            cwd / "uncheatable-eval" / "data",
            cwd / "raw" / "uncheatable-eval" / "latest",
            cwd / "raw" / "uncheatable_eval" / "latest",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_dir():
            return candidate

    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CACHE_DIR.resolve()


def load_uncheatable_eval(
    dataset: str,
    data_root: Optional[str] = None,
    max_documents: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
    **_,
) -> dict:
    """Load Uncheatable Eval documents for lm-evaluation-harness."""

    root = _resolve_data_root(data_root)
    files = _find_dataset_files(dataset, root)
    if not files:
        download_error: Optional[Exception] = None
        if AUTO_DOWNLOAD_ENABLED:
            try:
                LOGGER.info(
                    "No local Uncheatable Eval files for '%s' in %s. Attempting download.",
                    dataset,
                    root,
                )
                _download_latest_dataset(dataset, root)
            except Exception as exc:  # noqa: BLE001 - propagate as informative error
                download_error = exc
                LOGGER.warning(
                    "Automatic download for '%s' failed: %s",
                    dataset,
                    exc,
                )
        files = _find_dataset_files(dataset, root)
        if not files:
            message = (
                f"No Uncheatable Eval files found for prefix '{dataset}' in {root}."
            )
            if AUTO_DOWNLOAD_ENABLED and download_error is not None:
                raise FileNotFoundError(
                    message
                    + " Automatic download from GitHub failed; inspect the logs above."
                ) from download_error
            if not AUTO_DOWNLOAD_ENABLED:
                message += (
                    " Automatic download is disabled. Set UNCHEATABLE_EVAL_DISABLE_DOWNLOAD=0"
                    " or download the dumps manually."
                )
            raise FileNotFoundError(message)

    records = list(_iter_dataset_records(files))
    if not records:
        raise ValueError(
            f"No usable records found for Uncheatable Eval dataset '{dataset}' in {root}."
        )

    dataset_obj = datasets.Dataset.from_list(records)

    if shuffle_seed is not None:
        dataset_obj = dataset_obj.shuffle(seed=shuffle_seed)

    if max_documents is not None:
        max_documents = int(max_documents)
        if max_documents < len(dataset_obj):
            dataset_obj = dataset_obj.select(range(max_documents))

    LOGGER.info(
        "Loaded %d documents for Uncheatable Eval dataset '%s' from %d files in %s",
        len(dataset_obj),
        dataset,
        len(files),
        root,
    )

    return {"test": dataset_obj}


def _find_dataset_files(dataset: str, root: Path) -> List[Path]:
    patterns = [
        f"{dataset}_*.jsonl.gz",
        f"{dataset}_*.jsonl",
        f"{dataset}_*.json",
    ]
    return list(chain.from_iterable(sorted(root.glob(pattern)) for pattern in patterns))


def _iter_dataset_records(files: Iterable[Path]) -> Iterator[dict[str, str]]:
    for file_path in files:
        if file_path.name.endswith(".jsonl.gz"):
            yield from _iter_jsonl(file_path, compression="gzip")
        elif file_path.suffix == ".jsonl":
            yield from _iter_jsonl(file_path)
        elif file_path.suffix == ".json":
            yield from _iter_json(file_path)
        else:
            LOGGER.warning("Skipping unsupported file %s", file_path)


def _iter_jsonl(
    file_path: Path, compression: Optional[str] = None
) -> Iterator[dict[str, str]]:
    opener = gzip.open if compression == "gzip" else open
    with opener(file_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            yield _normalize_record(raw, file_path)


def _iter_json(file_path: Path) -> Iterator[dict[str, str]]:
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(
            f"Unexpected payload in {file_path}: expected a list, found {type(payload).__name__}."
        )

    for raw in payload:
        yield _normalize_record(raw, file_path)


def _normalize_record(raw: Any, file_path: Path) -> dict[str, str]:
    text = _extract_text(raw)
    if text is None or not str(text).strip():
        raise ValueError(f"Record in {file_path} does not contain text.")
    return {"text": str(text)}


def _extract_text(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        value = raw.get("text")
        if isinstance(value, str) and value.strip():
            return value
        for key in TEXT_FIELD_CANDIDATES:
            candidate = raw.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        for key in TEXT_FIELD_CANDIDATES:
            joined = _join_list_field(raw.get(key))
            if joined:
                return joined
        for key in LIST_FIELD_CANDIDATES:
            joined = _join_list_field(raw.get(key))
            if joined:
                return joined
        title = raw.get("title")
        body = raw.get("body")
        if isinstance(title, str) and isinstance(body, str):
            combined = f"{title.strip()}\n\n{body.strip()}".strip()
            if combined:
                return combined
        if isinstance(title, str) and title.strip():
            return title
        return json.dumps(raw, ensure_ascii=False)
    return str(raw)


def _join_list_field(value: Any) -> Optional[str]:
    if isinstance(value, list):
        text_items = [str(item) for item in value if item is not None]
        if text_items:
            return "\n".join(text_items)
    return None


def _download_latest_dataset(dataset: str, root: Path) -> List[Path]:
    owner = os.getenv("UNCHEATABLE_EVAL_REPO_OWNER", "Jellyfish042")
    repo = os.getenv("UNCHEATABLE_EVAL_REPO_NAME", "uncheatable_eval")
    data_path = os.getenv("UNCHEATABLE_EVAL_DATA_PATH", "data")
    branch = os.getenv("UNCHEATABLE_EVAL_BRANCH", "master")

    session = _github_session()
    headers = _github_headers()
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{data_path}"

    response = session.get(
        url, headers=headers, params={"ref": branch}, timeout=GITHUB_REQUEST_TIMEOUT
    )
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(
            f"Unexpected response from GitHub when listing {owner}/{repo}/{data_path}: {payload!r}"
        )

    candidate = _select_latest_entry(payload, dataset)
    if candidate is None:
        raise FileNotFoundError(
            f"Could not locate remote dump for '{dataset}' in {owner}/{repo}/{data_path}."
        )

    download_url = candidate.get("download_url")
    if not isinstance(download_url, str):
        raise ValueError(
            f"Entry for {candidate.get('name')} is missing a download_url field."
        )

    root.mkdir(parents=True, exist_ok=True)
    target_path = root / candidate["name"]
    if target_path.exists():
        LOGGER.info("Found previously downloaded file %s", target_path)
        return [target_path]

    LOGGER.info(
        "Downloading latest Uncheatable Eval dump for '%s' from %s",
        dataset,
        download_url,
    )
    download_headers = headers.copy()
    # For raw file downloads ensure we accept the content as-is.
    download_headers.pop("Accept", None)
    response = session.get(
        download_url, headers=download_headers, timeout=GITHUB_REQUEST_TIMEOUT
    )
    response.raise_for_status()

    if target_path.suffix == ".gz":
        target_path.write_bytes(response.content)
    else:
        target_path.write_text(response.text, encoding="utf-8")

    LOGGER.info("Wrote %s", target_path)
    return [target_path]


def _select_latest_entry(entries: List[dict[str, Any]], dataset: str) -> Optional[dict]:
    best_entry: Optional[dict] = None
    best_key: Optional[tuple[str, str, str]] = None
    prefix = f"{dataset}_"
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str) or not name.startswith(prefix):
            continue
        match = FILENAME_PATTERN.match(name)
        if match is None:
            continue
        end = match.group("end")
        start = match.group("start")
        key = (end, start, name)
        if best_key is None or key > best_key:
            best_entry = entry
            best_key = key
    return best_entry


def _github_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _github_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("UNCHEATABLE_EVAL_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


__all__ = ["load_uncheatable_eval"]
