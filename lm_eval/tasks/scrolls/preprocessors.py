from functools import partial

from transformers import AutoTokenizer

def _num_cpu_cores():
    # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python/55423170#55423170
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except ImportError:
        import os
        return len(os.sched_getaffinity(0))

def process_docs(dataset, custom_process=None, PRUNE_TOKENIZERS=[], PRUNE_MAX_TOKENS=4096, PRUNE_NUM_PROC=_num_cpu_cores()):

    def _drop_duplicates_in_input(untokenized_dataset):
        # from scrolls/evaluator/dataset_evaluator.py

        indices_to_keep = []
        id_to_idx = {}
        outputs = []
        for i, (id_, output) in enumerate(zip(untokenized_dataset["id"], untokenized_dataset["output"])):
            if id_ in id_to_idx:
                outputs[id_to_idx[id_]].append(output)
                continue
            indices_to_keep.append(i)
            id_to_idx[id_] = len(outputs)
            outputs.append([output])
        untokenized_dataset = untokenized_dataset.select(indices_to_keep).flatten_indices()
        untokenized_dataset = untokenized_dataset.remove_columns("output")
        untokenized_dataset = untokenized_dataset.add_column("outputs", outputs)
        return untokenized_dataset

    dataset = _drop_duplicates_in_input(dataset)
    if custom_process is not None:
        dataset = dataset.map(custom_process)
    
    if len(PRUNE_TOKENIZERS) > 0:
        tokenizers = [AutoTokenizer.from_pretrained(tokenizer) for tokenizer in PRUNE_TOKENIZERS]
        cache = {}

        def _get_prune_text(doc):
            return doc_to_text(doc)

        def _filter(sample):
            text = _get_prune_text(sample)
            cached = cache.get(text, None)
            if cached is None:
                for tokenizer in tokenizers:
                    if len(tokenizer(text).input_ids) > PRUNE_MAX_TOKENS:
                        cache[text] = False
                        return False
                cache[text] = True
                return True
            else:
                return cached

        dataset = dataset.filter(_filter, num_proc=PRUNE_NUM_PROC)

    return dataset

def _doc_prepended_question(doc):
    # "When a query is given in addition to the raw text (as
    # in QMSum, Qasper, NarrativeQA, QuALITY, and ContractNLI),
    # we prepend it to the text, using two newlines as a natural separator"
    input = doc["input"]
    split = input.find("\n\n")
    return {
        "id": doc["id"],
        "pid": doc["pid"],
        "input": input,
        "outputs": doc["outputs"],
        "question": input[0:split],
        "text": input[split + 2:]
    }

process_docs_prepended_question = partial(process_docs, custom_process=_doc_prepended_question)