# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import itertools  # noqa: I001
import random
from functools import cache

import datasets
import requests
from tqdm import tqdm

from lm_eval.tasks.ruler.common_utils import DEFAULT_SEQ_LENGTHS, get_tokenizer, get_limit_factor

CONFIG = {
    "tokens_to_generate": 32,
    "template": """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}""",
    "answer_prefix": """Answer:""",
}
SEED = 42
TEMPLATE = CONFIG["template"]
DOCUMENT_PROMPT = "Document {i}:\n{document}"


@cache
def download_json(url) -> dict:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data


@cache
def read_squad(
    url="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
) -> tuple[list[dict], list[str]]:
    """
    Read SQuAD dataset with fallback to HuggingFace datasets for CI environments.
    
    First attempts to download from the original SQuAD URL. If that fails (e.g., due to
    network restrictions in CI), falls back to loading from HuggingFace datasets.
    """
    try:
        # Try original URL first
        data = download_json(url)
        print(f"Successfully loaded SQuAD from original URL: {url}")
    except Exception as e:
        print(f"Failed to download SQuAD from original URL ({e}), falling back to HuggingFace datasets")
        
        try:
            # Fallback to HuggingFace datasets
            dataset = datasets.load_dataset("squad_v2", split="validation")
            
            # Convert HuggingFace dataset to expected format
            # HF format: {'context': str, 'answers': {'text': [...], 'answer_start': [...]}}
            # Original format: nested structure with data.data.paragraphs.qas
            data = {"data": []}
            contexts_seen = {}
            
            for item in dataset:
                # Only process items that have answers (not impossible questions)
                if item["answers"]["text"]:  # Skip impossible questions
                    context = item["context"]
                    
                    # Group by context to match original structure
                    if context not in contexts_seen:
                        contexts_seen[context] = {
                            "title": item["title"],
                            "paragraphs": [{"context": context, "qas": []}]
                        }
                        data["data"].append(contexts_seen[context])
                    
                    # Add QA to the appropriate context
                    qa_entry = {
                        "question": item["question"],
                        "answers": [{"text": text} for text in item["answers"]["text"]],
                        "is_impossible": False,
                        "id": item["id"]
                    }
                    contexts_seen[context]["paragraphs"][0]["qas"].append(qa_entry)
            
            print(f"Successfully loaded {len(dataset)} samples from HuggingFace datasets, converted to {len(data['data'])} context groups")
            
        except Exception as hf_error:
            raise Exception(f"Both original URL and HuggingFace fallback failed. "
                          f"Original error: {e}. HuggingFace error: {hf_error}")
    
    # Process data into expected format (same logic as before)
    total_docs = [p["context"] for d in data["data"] for p in d["paragraphs"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data["data"]:
        more_docs = [total_docs_dict[p["context"]] for p in d["paragraphs"]]
        for p in d["paragraphs"]:
            for qas in p["qas"]:
                if not qas["is_impossible"]:
                    total_qas.append(
                        {
                            "query": qas["question"],
                            "outputs": [a["text"] for a in qas["answers"]],
                            "context": [total_docs_dict[p["context"]]],
                            "more_context": [
                                idx
                                for idx in more_docs
                                if idx != total_docs_dict[p["context"]]
                            ],
                        }
                    )

    return total_qas, total_docs


@cache
def read_hotpotqa(
    url="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
) -> tuple[list[dict], list[str]]:
    """
    Read HotpotQA dataset with fallback to HuggingFace datasets for CI environments.
    
    First attempts to download from the original CMU URL. If that fails (e.g., due to
    network restrictions in CI), falls back to loading from HuggingFace datasets.
    """
    try:
        # Try original URL first
        data = download_json(url)
        print(f"Successfully loaded HotpotQA from original URL: {url}")
    except Exception as e:
        print(f"Failed to download HotpotQA from original URL ({e}), falling back to HuggingFace datasets")
        
        try:
            # Fallback to HuggingFace datasets
            dataset = datasets.load_dataset("hotpot_qa", "distractor", split="validation")
            
            # Convert HuggingFace dataset to expected format
            data = []
            for item in dataset:
                # HuggingFace format: context = {'title': [...], 'sentences': [[...], [...]]}
                # Original format: context = [[title, sentences], [title, sentences], ...]
                context_converted = []
                titles = item["context"]["title"]
                sentences = item["context"]["sentences"]
                
                for title, sentence_list in zip(titles, sentences):
                    context_converted.append([title, sentence_list])
                
                data.append({
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "supporting_facts": item["supporting_facts"],
                    "context": context_converted
                })
            
            print(f"Successfully loaded {len(data)} samples from HuggingFace datasets")
            
        except Exception as hf_error:
            raise Exception(f"Both original URL and HuggingFace fallback failed. "
                          f"Original error: {e}. HuggingFace error: {hf_error}")
    
    # Process data into expected format (same logic as before)
    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append(
            {
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": [
                    total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]
                ],
            }
        )

    return total_qas, total_docs


def generate_input_output(
    index: int, num_docs: int, qas: list[dict], docs: list[str]
) -> tuple[str, list[str]]:
    curr_q: str = qas[index]["query"]
    curr_a: list[str] = qas[index]["outputs"]
    curr_docs: list[int] = qas[index]["context"]
    curr_more: list[int] = qas[index].get("more_context", [])
    if num_docs < len(docs):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [
                i for i, d in enumerate(docs) if i not in curr_docs + curr_more
            ]
            all_docs = (
                curr_docs
                + curr_more
                + random.sample(
                    addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more))
                )
            )
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))

        all_docs = [docs[idx] for idx in all_docs]
    else:
        all_docs = docs

    random.Random(SEED).shuffle(all_docs)

    context = "\n\n".join(
        [DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)]
    )
    input_text = TEMPLATE.format(context=context, query=curr_q)
    return input_text, curr_a


def generate_samples(
    tokenizer,
    docs: list[str],
    qas: list[dict],
    max_seq_length: int,
    num_samples: int = 500,
    tokens_to_generate: int = 32,
    pre_samples: int = 0,
    incremental: int = 10,
    remove_newline_tab=False,
) -> list[dict]:
    write_jsons = []
    tokens_to_generate = tokens_to_generate

    # Find the perfect num_docs
    num_docs = incremental

    total_tokens = 0  # Track the total tokens generated for this example
    while total_tokens + tokens_to_generate < max_seq_length:
        input_text, answer = generate_input_output(0, num_docs, qas=qas, docs=docs)
        # Calculate the number of tokens in the example
        total_tokens = len(tokenizer(input_text + f" {answer}").input_ids)
        # print(
        #     f"Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Docs: {num_docs}"
        # )
        if total_tokens + tokens_to_generate > max_seq_length:
            num_docs -= incremental
            break

        num_docs += incremental
        if num_docs > len(docs):
            num_docs = len(docs)
            break
    # print("Number of documents:", num_docs)

    # Generate samples
    for index in tqdm(
        range(num_samples), desc=f"Generating QA Samples | {max_seq_length}"
    ):
        used_docs = num_docs
        while True:
            try:
                input_text, answer = generate_input_output(
                    index + pre_samples, used_docs, qas=qas, docs=docs
                )
                length = len(tokenizer(input_text).input_ids) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:  # noqa: E722
                if used_docs > incremental:
                    used_docs -= incremental

        if remove_newline_tab:
            input_text = " ".join(
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )

        formatted_output = {
            "index": index,
            "input": input_text,
            "outputs": answer,
            "length": length,
            "max_length": max_seq_length,
            "gen_prefix": "Answer:",
        }
        write_jsons.append(formatted_output)

    return write_jsons


def get_dataset(pretrained, docs, qas, max_seq_length=None, num_samples=500, **kwargs) -> list[dict]:
    tokenizer = get_tokenizer(pretrained)
    write_jsons = generate_samples(
        tokenizer=tokenizer,
        docs=docs,
        qas=qas,
        num_samples=num_samples,
        tokens_to_generate=32,
        max_seq_length=max_seq_length,
    )
    return write_jsons


def get_qa_dataset(ds, **kwargs) -> dict[str, datasets.Dataset]:
    pretrained = kwargs.get("tokenizer", kwargs.get("pretrained", {}))
    base_samples = kwargs.pop("num_samples_per_length", 500)  # Base sample count
    limit_factor = get_limit_factor(kwargs)
    num_samples = int(base_samples * limit_factor)  # Apply limit per sequence length
    if ds == "squad":
        qas, docs = read_squad()
    else:
        qas, docs = read_hotpotqa()
    df = (
        get_dataset(pretrained=pretrained, docs=docs, qas=qas, max_seq_length=seq, num_samples=num_samples)
        for seq in kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    )

    return {
        "test": datasets.Dataset.from_list(
            list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST
        )
    }


def get_squad(**kwargs):
    return get_qa_dataset("squad", **kwargs)


def get_hotpotqa(**kwargs):
    return get_qa_dataset("hotpotqa", **kwargs)
