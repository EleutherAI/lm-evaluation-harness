import inspect
import logging
from typing import Any

from lm_eval import utils as lm_eval_utils


eval_logger = logging.getLogger(__name__)


def _extract_task_runtime_config() -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Best-effort access to task config from the process_docs call stack."""
    frame = inspect.currentframe()
    while frame is not None:
        task_obj = frame.f_locals.get("self")
        config = getattr(task_obj, "config", None)
        if config is not None:
            metadata = getattr(config, "metadata", {}) or {}
            generation_kwargs = getattr(config, "generation_kwargs", {}) or {}
            doc_to_text = getattr(config, "doc_to_text", None)
            return metadata, generation_kwargs, doc_to_text
        frame = frame.f_back
    return {}, {}, None


def _resolve_max_model_len(metadata: dict[str, Any]) -> int | None:
    keys = (
        "max_model_len",
        "max_length",
        "context_length",
        "model_max_length",
    )
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            max_len = int(value)
            if max_len > 0:
                return max_len
        except (TypeError, ValueError):
            continue
    return None


def _resolve_tokenizer_name(metadata: dict[str, Any]) -> str | None:
    for key in ("tokenizer", "tokenizer_name", "pretrained", "model"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def process_docs(dataset):
    metadata, generation_kwargs, doc_to_text = _extract_task_runtime_config()

    max_model_len = _resolve_max_model_len(metadata)
    tokenizer_name = _resolve_tokenizer_name(metadata)
    if max_model_len is None:
        eval_logger.warning(
            "longbench2: unable to resolve model max length from metadata; skipping filter."
        )
        return dataset
    if not tokenizer_name:
        eval_logger.warning(
            "longbench2: unable to resolve tokenizer/pretrained name; skipping filter."
        )
        return dataset

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as exc:
        eval_logger.warning(
            "longbench2: failed to load tokenizer '%s' (%s); skipping filter.",
            tokenizer_name,
            exc,
        )
        return dataset

    fallback_max_gen_toks = int(generation_kwargs.get("max_gen_toks", 0) or 0)
    skip_count = 0

    def _keep_doc(doc: dict[str, Any]) -> bool:
        nonlocal skip_count
        prompt = (
            lm_eval_utils.apply_template(doc_to_text, doc)
            if isinstance(doc_to_text, str)
            else doc.get("context", "")
        )
        input_toks = len(tokenizer.encode(prompt, add_special_tokens=False))
        output_toks = int(doc.get("max_new_tokens", fallback_max_gen_toks) or 0)
        should_skip = input_toks > max_model_len or (
            input_toks + output_toks > max_model_len
        )
        if should_skip:
            skip_count += 1
            return False
        return True

    filtered_dataset = dataset.filter(_keep_doc)
    if skip_count:
        eval_logger.info(
            "longbench2: skipped %d prompts that exceed model context constraints.",
            skip_count,
        )
    return filtered_dataset