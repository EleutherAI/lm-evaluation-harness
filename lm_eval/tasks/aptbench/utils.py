from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datasets

from lm_eval.api.task import ConfigurableTask


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


eval_logger = logging.getLogger(__name__)

APTBENCH_REPO = "TencentYoutuResearch/APTBench"
APTBENCH_COMMIT = "9aab1301ab2c743be6bf11d1a2f3c3c6bc123ffc"

OPTIONAL_METADATA_FIELDS: tuple[str, ...] = (
    "ability",
    "source",
    "instance_id",
    "language",
    "model_type",
    "qid",
)

_DOWNLOAD_MANAGER: datasets.DownloadManager | None = None


def extract_answer(response: str) -> str | None:
    """Extract single-letter option before ')' or newline (matches predict.py)."""
    # Upstream quirk preserved for parity with APTBench/code/predict.py:
    # only matches a single letter followed immediately by ')' or '\n' (no EOF match).
    response = response.replace("*", "")
    match = re.search(r"\b([A-Za-z])[\)\n]", response)
    return match.group(1) if match else None


def extract_answer_action(response: str) -> str:
    """Extract first command/line up to newline or semicolon (matches predict.py)."""
    # Upstream quirk preserved for parity with APTBench/code/predict.py:
    # splits on ';' only when '\n' is present in the response.
    pattern = "\n"
    if pattern in response:
        return response.split(pattern, 1)[0].split(";", 1)[0]
    return response


def extract_answer_summ_ans(response: str) -> str | None:
    """Extract answer substring before closing ']' (matches predict.py)."""
    pattern = "]"
    if pattern in response:
        return response.split(pattern, 1)[0]
    return None


def extract_answer_cite(response: str) -> str | None:
    """Extract citation choice list before closing ')' (matches predict.py)."""
    pattern = ")"
    if pattern in response:
        return response.split(pattern, 1)[0]
    return None


def extract_answer_choice_ae(response: str) -> str | None:
    """Extract A-E choice letter, case-insensitive (matches predict.py)."""
    response = response.replace("*", "")
    match = re.search(r"\b([A-E])[\)\n\s:]", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-E])\s*$", response, re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_answer_choice_ad(response: str) -> str | None:
    """Extract A-D choice letter, case-insensitive (matches predict.py)."""
    response = response.replace("*", "")
    match = re.search(r"\b([A-D])[\)\n\s:]", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-D])\s*$", response, re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_answer_true_false(response: str) -> str | None:
    """Extract 'correct' or 'wrong' judgment (matches predict.py)."""
    response = response.lower().replace("*", "")
    match = re.search(r"\b(correct|wrong)\b", response)
    return match.group(1) if match else None


TASK_SPECS: dict[tuple[str, str], dict[str, Any]] = {
    ("agentic_math", "planning_single"): {
        "template": "math_planning_single_3shot.txt",
        "prompt_fields": {"$REPO_INFO$": "repo_info", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "extractor": extract_answer_choice_ae,
        "judge": "eq",
        "metric": "acc",
    },
    ("agentic_math", "feedback_tf"): {
        "template": "math_feedback_tf.txt",
        # Upstream quirk preserved for parity with APTBench/code/predict.py:
        # "$CHOICES$" is unused in the default template but retained for custom prompt_dir overrides.
        "prompt_fields": {"$REPO_INFO$": "repo_info", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "extractor": extract_answer_true_false,
        "judge": "eq",
        "metric": "acc",
    },
    ("agentic_math", "action_cal"): {
        "template": "math_action_calculation.txt",
        "prompt_fields": {"$REPO_INFO$": "repo_info", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "extractor": extract_answer_choice_ad,
        "judge": "eq",
        "metric": "acc",
    },
    ("env_setup", "plan"): {
        "template": "env_setup_plan_3shot.txt",
        "prompt_fields": {"$REPO_INFO$": "repo_info", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "extractor": extract_answer,
        "judge": "eq",
        "metric": "acc",
    },
    ("env_setup", "action"): {
        "template": "env_setup_action_3shot.txt",
        "prompt_fields": {
            "$EXE_PLAN$": "execution_plan",
            "$EXE_CMDS$": "executed_cmds",
        },
        "answer_key": "target_cmd",
        "extractor": extract_answer_action,
        "judge": "eq",
        "metric": "exact_match",
    },
    ("env_setup", "error"): {
        "template": "env_setup_error_3shot.txt",
        "prompt_fields": {
            "$SETUP$": "setup_instruct",
            "$ISSUE_TITLE$": "issue_title",
            "$ISSUE_BODY$": "issue_body",
            "$CHOICES$": "choices",
        },
        "answer_key": "answer",
        "extractor": extract_answer,
        "judge": "eq",
        "metric": "acc",
    },
    ("deepresearch", "plan_zh"): {
        "template": "deepresearch_plan_zh_3shot.txt",
        "prompt_fields": {
            "$QUERY$": "query",
            "$TRAJ$": "trajectory",
            "$CHOICES$": "choices",
        },
        "answer_key": "answer",
        "extractor": extract_answer,
        "judge": "eq",
        "metric": "acc",
    },
    ("deepresearch", "plan_en"): {
        "template": "deepresearch_plan_en_3shot.txt",
        "prompt_fields": {
            "$QUERY$": "query",
            "$TRAJ$": "trajectory",
            "$CHOICES$": "choices",
        },
        "answer_key": "answer",
        "extractor": extract_answer,
        "judge": "eq",
        "metric": "acc",
    },
    ("deepresearch", "summ_ans_zh"): {
        "template": "deepresearch_summ_ans_zh_3shot.txt",
        "prompt_fields": {"$QUERY$": "query", "$TRAJ$": "trajectory"},
        "answer_key": "answer",
        "extractor": extract_answer_summ_ans,
        "judge": "summ_ans",
        "metric": "exact_match",
    },
    ("deepresearch", "summ_ans_en"): {
        "template": "deepresearch_summ_ans_en_3shot.txt",
        "prompt_fields": {"$QUERY$": "query", "$TRAJ$": "trajectory"},
        "answer_key": "answer",
        "extractor": extract_answer_summ_ans,
        "judge": "summ_ans",
        "metric": "exact_match",
    },
    ("deepresearch", "openend_plan_en"): {
        "template": "deepresearch_openend_plan_en_3shot.txt",
        "prompt_fields": {"$QUERY$": "question", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "extractor": extract_answer,
        "judge": "eq",
        "metric": "acc",
    },
    ("deepresearch", "openend_citation_en"): {
        "template": "deepresearch_openend_citation_en_3shot.txt",
        "prompt_fields": {
            "$ARTICLE$": "article",
            "$CHOICES$": "choices",
            "$WEB_PAGE$": "url_content",
        },
        "answer_key": "answer",
        "extractor": extract_answer_cite,
        "judge": "cite_set",
        "metric": "acc",
    },
    ("deepresearch", "openend_citation_zh"): {
        "template": "deepresearch_openend_citation_zh_3shot.txt",
        "prompt_fields": {
            "$ARTICLE$": "article",
            "$CHOICES$": "choices",
            "$WEB_PAGE$": "url_content",
        },
        "answer_key": "answer",
        "extractor": extract_answer_cite,
        "judge": "cite_set",
        "metric": "acc",
    },
    ("deepresearch", "openend_quality_en"): {
        "template": "deepresearch_openend_quality_en_2shot.txt",
        "prompt_fields": {"$QUERY$": "query", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "extractor": extract_answer,
        "judge": "eq",
        "metric": "acc",
    },
    ("deepresearch", "openend_quality_zh"): {
        "template": "deepresearch_openend_quality_zh_2shot.txt",
        "prompt_fields": {"$QUERY$": "query", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "extractor": extract_answer,
        "judge": "eq",
        "metric": "acc",
    },
    ("issue_fix", "fix_patch"): {
        "template": "issue_fix_fix_patch_3shot.txt",
        "prompt_fields": {"$ISSUE$": "issue_statement", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "extractor": extract_answer,
        "judge": "eq",
        "metric": "acc",
    },
    ("issue_fix", "locate"): {
        "template": "issue_fix_locate_3shot.txt",
        "prompt_fields": {"$ISSUE$": "issue_statement", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "extractor": extract_answer,
        "judge": "eq",
        "metric": "acc",
    },
    ("issue_fix", "plan"): {
        "template": "issue_fix_plan_3shot.txt",
        "prompt_fields": {"$TRAJ$": "trajs", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "extractor": extract_answer,
        "judge": "eq",
        "metric": "acc",
    },
    ("issue_fix", "action"): {
        "template": "issue_fix_action_3shot.txt",
        "prompt_fields": {"$TRAJ$": "trajs"},
        "answer_key": "answer",
        "extractor": extract_answer_action,
        "judge": "eq",
        "metric": "exact_match",
    },
    ("issue_fix", "test_patch"): {
        "template": "issue_fix_test_patch_3shot.txt",
        "prompt_fields": {"$PROBLEM$": "problem_statement", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "extractor": extract_answer,
        "judge": "eq",
        "metric": "acc",
    },
    ("tool", "acebench_api_select"): {
        "template": "tool_acebench_api_select_3shot.txt",
        "prompt_fields": {"$USER_PROMPT$": "user_prompt", "$FUNCS$": "function"},
        "answer_key": "answer4select",
        "extractor": extract_answer_action,
        "judge": "eq",
        "metric": "exact_match",
    },
    ("tool", "acebench_api_param"): {
        "template": "tool_acebench_api_param_3shot.txt",
        "prompt_fields": {
            "$USER_PROMPT$": "user_prompt",
            "$FUNCS$": "function",
            "$ANSWER_SELECT$": "answer4select",
            "$PARA_NAME$": "param_name",
        },
        "answer_key": "answer4param",
        "extractor": extract_answer_action,
        "judge": "eq",
        "metric": "exact_match",
    },
    ("tool", "bfcl_v4_api_select"): {
        "template": "tool_bfcl_v4_api_select_3shot.txt",
        "prompt_fields": {"$USER_PROMPT$": "user_prompt", "$FUNCS$": "function"},
        "answer_key": "answer4select",
        "extractor": extract_answer_action,
        "judge": "eq",
        "metric": "exact_match",
    },
    ("tool", "bfcl_v4_api_param"): {
        "template": "tool_bfcl_v4_api_param_3shot.txt",
        "prompt_fields": {
            "$USER_PROMPT$": "user_prompt",
            "$FUNCS$": "function",
            "$ANSWER_SELECT$": "answer4select",
            "$PARA_NAME$": "param_name",
        },
        "answer_key": "answer4param",
        "extractor": extract_answer_action,
        "judge": "eq",
        "metric": "exact_match",
    },
}


def _get_download_manager() -> datasets.DownloadManager:
    """Return a lazily initialized shared datasets.DownloadManager instance."""
    global _DOWNLOAD_MANAGER
    if _DOWNLOAD_MANAGER is None:
        _DOWNLOAD_MANAGER = datasets.DownloadManager()
    return _DOWNLOAD_MANAGER


def _resolve_source_options(
    kwargs: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None, str]:
    """Resolve namespaced APTBench source overrides from metadata or environment variables.

    Only ``aptbench_*`` keys are read from ``kwargs`` so standard model backend
    arguments (such as ``revision`` or ``data_dir`` merged via ``--model_args``)
    cannot collide with APTBench dataset/prompt resolution.
    """
    aptbench_dir = kwargs.get("aptbench_dir") or os.environ.get("APTBENCH_DIR") or None
    data_dir = (
        kwargs.get("aptbench_data_dir") or os.environ.get("APTBENCH_DATA_DIR") or None
    )
    prompt_dir = (
        kwargs.get("aptbench_prompt_dir")
        or os.environ.get("APTBENCH_PROMPT_DIR")
        or None
    )
    revision = (
        kwargs.get("aptbench_revision")
        or os.environ.get("APTBENCH_REVISION")
        or APTBENCH_COMMIT
    )
    return aptbench_dir, data_dir, prompt_dir, revision


def _download_remote_file(rel_path: str, revision: str) -> Path:
    """Download a file from the pinned APTBench GitHub repository via datasets cache."""
    url = f"https://raw.githubusercontent.com/{APTBENCH_REPO}/{revision}/{rel_path}"
    try:
        downloaded = _get_download_manager().download(url)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download APTBench file from '{url}'. "
            "For offline or local evaluation, pass "
            '--metadata \'{"aptbench_dir": "/path/to/APTBench"}\' '
            f"or set the APTBENCH_DIR environment variable. Original error: {exc}"
        ) from exc
    return Path(downloaded)


def _resolve_data_file(
    aptbench_dataset: str,
    aptbench_task: str,
    *,
    data_dir: str | None,
    aptbench_dir: str | None,
    revision: str,
) -> Path:
    """Locate local JSONL data file or download it from GitHub."""
    if data_dir:
        path = Path(data_dir) / aptbench_dataset / aptbench_task / "input_data.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"APTBench data file not found: {path}")
        return path
    if aptbench_dir:
        path = (
            Path(aptbench_dir)
            / "data"
            / aptbench_dataset
            / aptbench_task
            / "input_data.jsonl"
        )
        if not path.is_file():
            raise FileNotFoundError(f"APTBench data file not found: {path}")
        return path
    return _download_remote_file(
        f"data/{aptbench_dataset}/{aptbench_task}/input_data.jsonl",
        revision,
    )


def _load_prompt_template(
    template_name: str,
    *,
    prompt_dir: str | None,
    aptbench_dir: str | None,
    revision: str,
) -> str:
    """Load an APTBench prompt template verbatim from local override or GitHub.

    Local files are always read directly from disk so edits made within the same
    process take effect immediately; remote files use datasets' on-disk cache.
    """
    if prompt_dir:
        path = Path(prompt_dir) / template_name
        if not path.is_file():
            raise FileNotFoundError(f"APTBench prompt file not found: {path}")
    elif aptbench_dir:
        path = Path(aptbench_dir) / "code" / "prompts" / template_name
        if not path.is_file():
            raise FileNotFoundError(f"APTBench prompt file not found: {path}")
    else:
        path = _download_remote_file(f"code/prompts/{template_name}", revision)

    return path.read_text(encoding="utf-8")


def load_aptbench_dataset(
    *,
    aptbench_dataset: str,
    aptbench_task: str,
    aptbench_dir: str | None = None,
    aptbench_data_dir: str | None = None,
    aptbench_revision: str = APTBENCH_COMMIT,
) -> dict[str, datasets.Dataset]:
    """Load and validate an APTBench task split into a clean HuggingFace Dataset."""
    key = (aptbench_dataset, aptbench_task)
    if key not in TASK_SPECS:
        raise ValueError(f"Unknown APTBench task: {key}")

    spec = TASK_SPECS[key]
    data_path = _resolve_data_file(
        aptbench_dataset,
        aptbench_task,
        data_dir=aptbench_data_dir,
        aptbench_dir=aptbench_dir,
        revision=aptbench_revision,
    )

    required_fields = list(dict.fromkeys(spec["prompt_fields"].values()))
    answer_key = spec["answer_key"]
    expected_fields = (*required_fields, answer_key)

    raw_items: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = [
                col for col in expected_fields if col not in item or item[col] is None
            ]
            if missing:
                raise ValueError(
                    f"APTBench data file '{data_path}' line {line_num} "
                    f"(uuid={item.get('uuid', 'N/A')!r}) is missing or has null "
                    f"value for required field(s): {', '.join(missing)}"
                )
            raw_items.append(item)

    if not raw_items:
        raise ValueError(f"APTBench data file '{data_path}' contains no records.")

    present_optional = [
        col
        for col in OPTIONAL_METADATA_FIELDS
        if any(item.get(col) is not None for item in raw_items)
    ]

    rows: list[dict[str, Any]] = []
    for item in raw_items:
        row: dict[str, Any] = {
            "uuid": str(item.get("uuid", "")),
            answer_key: item[answer_key],
        }
        for col in required_fields:
            row[col] = item[col]
        for col in present_optional:
            val = item.get(col)
            row[col] = "" if val is None else str(val)
        rows.append(row)

    return {"test": datasets.Dataset.from_list(rows)}


def format_prompt(
    template: str, doc: Mapping[str, Any], spec: Mapping[str, Any]
) -> str:
    """Format a document using a raw prompt template and sequential str.replace."""
    prompt = template
    for placeholder, data_key in spec["prompt_fields"].items():
        if data_key in doc and doc[data_key] is not None:
            prompt = prompt.replace(placeholder, str(doc[data_key]).strip())
        else:
            eval_logger.warning(
                "APTBench doc %r is missing required prompt field %r for placeholder %r.",
                doc.get("uuid", "N/A"),
                data_key,
                placeholder,
            )
    return prompt


def format_target(doc: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    """Return the ground-truth answer string for an APTBench document."""
    ans = doc[spec["answer_key"]]
    if isinstance(ans, list):
        return ",".join(str(x) for x in ans)
    return str(ans)


def score_response(
    doc: Mapping[str, Any], results: list[str], spec: Mapping[str, Any]
) -> dict[str, float]:
    """Extract prediction and compute APTBench ACC / EM metrics for one document."""
    extractor: Callable[[str], str | None] = spec["extractor"]

    raw_output = results[0] if results else ""
    if not raw_output:
        pred = None
    else:
        output = raw_output.strip()
        pred = extractor(output)
        if pred is not None:
            pred = pred.strip()

    answer = doc[spec["answer_key"]]
    judge_mode = spec["judge"]

    if judge_mode == "cite_set":
        # Upstream quirk preserved for parity with APTBench/code/predict.py:
        # splits comma-separated predictions without stripping whitespace per element.
        score = float(set(answer) == (set(pred.split(",")) if pred else set()))
        return {"score": score, "acc": score}

    if judge_mode == "summ_ans":
        strict = float(pred == answer)
        flex = float(
            bool(pred) and (pred.strip('"') == answer or pred.strip("'") == answer)
        )
        return {
            "score": strict,
            "exact_match": strict,
            "exact_match_flex": flex,
        }

    score = float(pred == answer)
    return {"score": score, spec["metric"]: score}


class APTBenchTask(ConfigurableTask):
    """ConfigurableTask subclass holding per-instance APTBench spec and template."""

    def __init__(
        self,
        data_dir: str | None = None,
        cache_dir: str | None = None,
        download_mode: datasets.DownloadMode | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        cfg = dict(config or {})
        cfg.pop("class", None)
        dataset_kwargs = cfg.get("dataset_kwargs") or {}
        aptbench_dataset = dataset_kwargs.get("aptbench_dataset")
        aptbench_task = dataset_kwargs.get("aptbench_task")
        if not aptbench_dataset or not aptbench_task:
            raise ValueError(
                "Both 'aptbench_dataset' and 'aptbench_task' must be specified in dataset_kwargs."
            )
        key = (aptbench_dataset, aptbench_task)
        if key not in TASK_SPECS:
            raise ValueError(f"Unknown APTBench task: {key}")

        self._aptbench_dataset = aptbench_dataset
        self._aptbench_task = aptbench_task
        self._spec = TASK_SPECS[key]
        self._prompt_template = ""

        # Mark process_results as handled by this subclass so ConfigurableTask.__init__
        # skips registry / HF evaluate lookups for custom metrics such as 'score'.
        cfg.setdefault("process_results", "process_results")
        super().__init__(
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
            config=cfg,
        )

    def download(
        self, dataset_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Load the prompt template and dataset for this task instance."""
        source_kwargs: dict[str, Any] = {
            **(self.config.metadata or {}),
            **(dataset_kwargs or self.config.dataset_kwargs or {}),
        }
        aptbench_dir, data_dir, prompt_dir, revision = _resolve_source_options(
            source_kwargs
        )
        self._prompt_template = _load_prompt_template(
            self._spec["template"],
            prompt_dir=prompt_dir,
            aptbench_dir=aptbench_dir,
            revision=revision,
        )
        self.dataset = load_aptbench_dataset(
            aptbench_dataset=self._aptbench_dataset,
            aptbench_task=self._aptbench_task,
            aptbench_dir=aptbench_dir,
            aptbench_data_dir=data_dir,
            aptbench_revision=revision,
        )

    def doc_to_text(self, doc: Mapping[str, Any], doc_to_text: Any = None) -> str:
        """Render the prompt string using this task's loaded template."""
        return format_prompt(self._prompt_template, doc, self._spec)

    def doc_to_target(self, doc: Mapping[str, Any], doc_to_target: Any = None) -> str:
        """Return the ground-truth answer string for this document."""
        return format_target(doc, self._spec)

    def process_results(
        self, doc: dict[str, Any], results: list[str]
    ) -> dict[str, float]:
        """Score a model response for a single document."""
        return score_response(doc, results, self._spec)
