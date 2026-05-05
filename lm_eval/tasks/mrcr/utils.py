import json
import logging
import math
import multiprocessing as mp
import os
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

import datasets
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

eval_logger = logging.getLogger(__name__)


DEFAULT_STOP_SEQUENCES = ["</s>", "<|im_end|>", "<|endoftext|>"]
DEFAULT_TOKENIZE_CHUNK_SIZE = 8
MRCR_BINS = [
    (0, 8192),
    (8192, 16384),
    (16384, 32768),
    (32768, 65536),
    (65536, 131072),
    (131072, 262144),
    (262144, 524288),
    (524288, 1048576),
]
MRCR_BIN_SCORE_METRICS = {
    8192: "score_gt_0_le_8k",
    16384: "score_gt_8k_le_16k",
    32768: "score_gt_16k_le_32k",
    65536: "score_gt_32k_le_64k",
    131072: "score_gt_64k_le_128k",
    262144: "score_gt_128k_le_256k",
    524288: "score_gt_256k_le_512k",
    1048576: "score_gt_512k_le_1m",
}
MRCR_BIN_COUNT_METRICS = {
    8192: "count_gt_0_le_8k",
    16384: "count_gt_8k_le_16k",
    32768: "count_gt_16k_le_32k",
    65536: "count_gt_32k_le_64k",
    131072: "count_gt_64k_le_128k",
    262144: "count_gt_128k_le_256k",
    524288: "count_gt_256k_le_512k",
    1048576: "count_gt_512k_le_1m",
}
_TOKENIZER_WORKER = None


def _tokenization_progress(total: int):
    return tqdm(
        total=total,
        desc="Tokenizing MRCR prompts",
        unit="doc",
    )


def grade_response(response: str, answer: str, random_string_to_prepend: str) -> float:
    """We follow grading logic from https://huggingface.co/datasets/openai/mrcr with
    filtering out the thinking tags if applicable to the response."""

    if "</think>" in response:
        response = response.split("</think>", 1)[1]

    response = response.strip()
    if not response.startswith(random_string_to_prepend):
        return 0.0

    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def _bin_max_for_tokens(num_prompt_tokens: int) -> int:
    for lower, upper in MRCR_BINS:
        if lower < num_prompt_tokens <= upper:
            return upper
    raise ValueError(
        f"MRCR prompt length {num_prompt_tokens} is outside supported bins."
    )


def process_results(
    doc: dict, results: list[str]
) -> dict[str, float | tuple[float, int]]:
    bin_max = _bin_max_for_tokens(int(doc["n_prompt_tokens"]))
    score = grade_response(
        results[0],
        doc["answer"],
        doc["random_string_to_prepend"],
    )
    result = {
        "score": score,
        "AUC": (score, bin_max),
    }
    result[MRCR_BIN_SCORE_METRICS[bin_max]] = score
    result[MRCR_BIN_COUNT_METRICS[bin_max]] = 1
    return result


def _power_of_2_generation_tokens(
    n_prompt_tokens: int,
    max_model_len: int,
) -> int:
    # Round the exact max_gen_tokens budget to a power-of-2 that fits within the
    # model's context length. Originally, each sample in MRCR has a different 
    # max_gen_tokens budget which is computed as max_model_len - n_prompt_tokens. 
    # However, this doesn't allow grouping of requests via num_concurrent arg as each 
    # sample ends up with different gen_kwargs. To enable grouping of requests, we use 
    # a discrete max_tokens budget which is computed as the next power-of-2 that fits 
    # within the model context. If the first larger power-of-2 doesn't fit within the 
    # model context, we use the first smaller power-of-2.
    remaining_tokens = max_model_len - n_prompt_tokens
    first_larger_bucket = 1 << (remaining_tokens - 1).bit_length()
    if n_prompt_tokens + first_larger_bucket <= max_model_len:
        return first_larger_bucket
    return max(1, first_larger_bucket // 2)


def infer_max_model_len(
    model_source: str | None,
    explicit_max_model_len: Any = None,
    fallback_max_length: Any = None,
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> int | None:
    if explicit_max_model_len is not None:
        return int(explicit_max_model_len)

    if fallback_max_length is not None:
        return int(fallback_max_length)

    if not model_source:
        return None

    try:
        cfg = AutoConfig.from_pretrained(
            model_source,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        if hasattr(cfg, "text_config") and hasattr(cfg, "vision_config"):
            return int(cfg.get_text_config().max_position_embeddings)
        return int(cfg.max_position_embeddings)
    except Exception as exc:
        eval_logger.warning(
            "Could not infer max_model_len for MRCR from %s: %s",
            model_source,
            exc,
        )
        return None


def _tokenize_lengths(tokenizer, texts: list[str]) -> list[int]:
    encodings = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return [len(ids) for ids in encodings["input_ids"]]


def _init_tokenizer_worker(
    tokenizer_source: str,
    tokenizer_init_kwargs: dict[str, Any],
):
    global _TOKENIZER_WORKER
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _TOKENIZER_WORKER = AutoTokenizer.from_pretrained(
        tokenizer_source,
        **tokenizer_init_kwargs,
    )


def _render_prompt_texts_worker(
    chunk: tuple[list[list[dict[str, Any]]], list[str]],
) -> tuple[list[str], list[tuple[int, int]]]:
    if _TOKENIZER_WORKER is None:
        raise RuntimeError("MRCR tokenizer worker was not initialized.")
    messages, answers = chunk
    return render_prompt_texts(_TOKENIZER_WORKER, messages, answers)


def _iter_tokenization_chunks(
    batched_messages: list[list[dict[str, Any]]],
    batched_expected_answer: list[str],
    chunk_size: int,
):
    for start in range(0, len(batched_messages), chunk_size):
        end = start + chunk_size
        yield batched_messages[start:end], batched_expected_answer[start:end]


def render_prompt_texts(
    tokenizer,
    batched_messages: list[list[dict[str, Any]]],
    batched_expected_answer: list[str],
) -> tuple[list[str], list[tuple[int, int]]]:
    prompt_texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in batched_messages
    ]
    prompt_and_answer_texts = [
        tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": answer}],
            tokenize=False,
            add_generation_prompt=False,
        )
        for messages, answer in zip(
            batched_messages,
            batched_expected_answer,
            strict=True,
        )
    ]

    prompt_lens = _tokenize_lengths(tokenizer, prompt_texts)
    prompt_and_answer_lens = _tokenize_lengths(tokenizer, prompt_and_answer_texts)
    answer_lens = [
        prompt_and_answer_len - prompt_len
        for prompt_and_answer_len, prompt_len in zip(
            prompt_and_answer_lens,
            prompt_lens,
            strict=True,
        )
    ]
    return prompt_texts, list(zip(prompt_lens, answer_lens, strict=True))


def render_prompt_texts_chunked(
    tokenizer,
    batched_messages: list[list[dict[str, Any]]],
    batched_expected_answer: list[str],
    chunk_size: int,
) -> tuple[list[str], list[tuple[int, int]]]:
    chunk_size = max(1, int(chunk_size))
    prompt_texts: list[str] = []
    counts: list[tuple[int, int]] = []
    chunks = _iter_tokenization_chunks(
        batched_messages,
        batched_expected_answer,
        chunk_size,
    )
    with _tokenization_progress(total=len(batched_messages)) as pbar:
        for chunk_messages, chunk_answers in chunks:
            chunk_prompt_texts, chunk_counts = render_prompt_texts(
                tokenizer,
                chunk_messages,
                chunk_answers,
            )
            prompt_texts.extend(chunk_prompt_texts)
            counts.extend(chunk_counts)
            pbar.update(len(chunk_messages))
    return prompt_texts, counts


def render_prompt_texts_parallel(
    tokenizer_source: str,
    tokenizer_init_kwargs: dict[str, Any],
    batched_messages: list[list[dict[str, Any]]],
    batched_expected_answer: list[str],
    num_workers: int,
    chunk_size: int,
) -> tuple[list[str], list[tuple[int, int]]]:
    num_workers = max(1, int(num_workers))
    chunk_size = max(1, int(chunk_size))
    chunks = list(
        _iter_tokenization_chunks(
            batched_messages,
            batched_expected_answer,
            chunk_size,
        )
    )
    if not chunks:
        return [], []

    eval_logger.info(
        "Tokenizing MRCR prompts with %s workers over %s chunks.",
        num_workers,
        len(chunks),
    )
    prompt_texts: list[str] = []
    counts: list[tuple[int, int]] = []
    with mp.Pool(
        processes=min(num_workers, len(chunks)),
        initializer=_init_tokenizer_worker,
        initargs=(tokenizer_source, tokenizer_init_kwargs),
    ) as pool:
        with _tokenization_progress(total=len(batched_messages)) as pbar:
            for chunk_prompt_texts, chunk_counts in pool.imap(
                _render_prompt_texts_worker,
                chunks,
            ):
                prompt_texts.extend(chunk_prompt_texts)
                counts.extend(chunk_counts)
                pbar.update(len(chunk_prompt_texts))
    return prompt_texts, counts


def _mean_scores_by_bin(items: list[tuple[float, int]]) -> dict[int, float]:
    grouped_scores: dict[int, list[float]] = defaultdict(list)
    for score, bin_max in items:
        grouped_scores[int(bin_max)].append(float(score))

    return {
        bin_max: sum(scores) / len(scores)
        for bin_max, scores in grouped_scores.items()
        if scores
    }

def AUC(items: list[tuple[float, int]]) -> float:
    points = sorted(_mean_scores_by_bin(items).items())
    if not points:
        return math.nan
    if len(points) == 1:
        return points[0][1] * 100.0

    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0

    width = points[-1][0] - points[0][0]
    if width <= 0:
        return points[-1][1] * 100.0
    return (area / width) * 100.0


def prepare_docs(
    raw_docs: list[dict[str, Any]],
    *,
    tokenizer=None,
    tokenizer_source: str | None = None,
    tokenizer_init_kwargs: dict[str, Any] | None = None,
    max_model_len: int | None,
    bin_l: int = -1,
    bin_h: int = -1,
    short_to_long: bool = False,
    num_eval_samples: int = -1,
    num_tokenize_workers: int = 1,
    tokenize_chunk_size: int = DEFAULT_TOKENIZE_CHUNK_SIZE,
) -> datasets.Dataset:
    batched_messages = [json.loads(doc["prompt"]) for doc in raw_docs]
    batched_answers = [doc["answer"] for doc in raw_docs]
    tokenizer_init_kwargs = tokenizer_init_kwargs or {}
    if max_model_len is None:
        raise ValueError(
            "MRCR requires max_model_len or max_length so infeasible examples can "
            "be filtered and generation budgets can be computed."
        )
    if int(num_tokenize_workers) > 1 and tokenizer_source:
        prompt_texts, counts = render_prompt_texts_parallel(
            tokenizer_source=tokenizer_source,
            tokenizer_init_kwargs=tokenizer_init_kwargs,
            batched_messages=batched_messages,
            batched_expected_answer=batched_answers,
            num_workers=int(num_tokenize_workers),
            chunk_size=int(tokenize_chunk_size),
        )
    else:
        if int(num_tokenize_workers) > 1 and not tokenizer_source:
            eval_logger.warning(
                "MRCR num_tokenize_workers=%s requires tokenizer_source; falling "
                "back to sequential tokenization.",
                num_tokenize_workers,
            )
        if tokenizer is None:
            if not tokenizer_source:
                raise ValueError(
                    "MRCR prepare_docs requires tokenizer or tokenizer_source."
                )
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                **tokenizer_init_kwargs,
            )
        prompt_texts, counts = render_prompt_texts_chunked(
            tokenizer,
            batched_messages,
            batched_answers,
            int(tokenize_chunk_size),
        )

    processed_docs = []
    for doc, messages, prompt_text, (n_prompt_tokens, n_expected_answer_tokens) in zip(
        raw_docs,
        batched_messages,
        prompt_texts,
        counts,
        strict=True,
    ):
        est_total_tokens = n_prompt_tokens + n_expected_answer_tokens
        if max_model_len is not None and est_total_tokens > max_model_len:
            continue
        if n_prompt_tokens >= max_model_len:
            continue
        if bin_l != -1 and bin_h != -1 and bin_h > bin_l:
            if not (bin_l < est_total_tokens <= bin_h):
                continue

        max_generation_tokens = _power_of_2_generation_tokens(
            n_prompt_tokens=n_prompt_tokens,
            max_model_len=max_model_len,
        )

        processed_docs.append(
            {
                **doc,
                "messages": messages,
                "prompt_text": prompt_text,
                "n_prompt_tokens": n_prompt_tokens,
                "n_expected_answer_tokens": n_expected_answer_tokens,
                "est_total_tokens": est_total_tokens,
                "max_generation_tokens": max_generation_tokens,
            }
        )

    processed_docs.sort(
        key=lambda doc: doc["n_prompt_tokens"],
        reverse=not short_to_long,
    )
    if num_eval_samples != -1:
        processed_docs = processed_docs[: int(num_eval_samples)]
    return datasets.Dataset.from_list(processed_docs)


def load_dataset(
    *,
    num_needles: int,
    model: str | None = None,
    pretrained: str | None = None,
    tokenizer: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    max_model_len: Any = None,
    max_length: Any = None,
    bin_l: int = -1,
    bin_h: int = -1,
    short_to_long: bool = False,
    num_eval_samples: int = -1,
    num_tokenize_workers: int = 1,
    tokenize_chunk_size: int = DEFAULT_TOKENIZE_CHUNK_SIZE,
    **_: Any,
) -> dict[str, datasets.Dataset]:
    tokenizer_source = tokenizer or model or pretrained
    config_source = model or pretrained or tokenizer_source
    if not tokenizer_source:
        raise ValueError(
            "MRCR requires `model=` in --model_args or `tokenizer` in metadata "
            "so prompt lengths can be computed."
        )
    if tokenizer:
        eval_logger.info("Using MRCR tokenizer override from metadata: %s", tokenizer)

    files = [
        f"{num_needles}needle/{num_needles}needle_0.parquet",
        f"{num_needles}needle/{num_needles}needle_1.parquet",
    ]
    local_files = [
        hf_hub_download(repo_id="openai/mrcr", filename=filename, repo_type="dataset")
        for filename in files
    ]
    raw_dataset = datasets.load_dataset(
        "parquet", data_files=local_files, split="train"
    )

    tokenizer_init_kwargs = {
        "trust_remote_code": trust_remote_code,
        "revision": revision,
    }
    task_tokenizer = None
    if int(num_tokenize_workers) <= 1:
        task_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            **tokenizer_init_kwargs,
        )
    effective_max_model_len = infer_max_model_len(
        model_source=config_source,
        explicit_max_model_len=max_model_len,
        fallback_max_length=max_length,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )

    processed = prepare_docs(
        list(raw_dataset),
        tokenizer=task_tokenizer,
        tokenizer_source=tokenizer_source,
        tokenizer_init_kwargs=tokenizer_init_kwargs,
        max_model_len=effective_max_model_len,
        bin_l=int(bin_l),
        bin_h=int(bin_h),
        short_to_long=bool(short_to_long),
        num_eval_samples=int(num_eval_samples),
        num_tokenize_workers=int(num_tokenize_workers),
        tokenize_chunk_size=int(tokenize_chunk_size),
    )
    return {"train": processed}
