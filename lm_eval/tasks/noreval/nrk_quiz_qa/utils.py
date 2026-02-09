import re
from lm_eval.api.samplers import ContextSampler


def parse_id(id_str):
    """
    Parse ID like '1.16561460-11' or '1.16561460-12b' into a sortable tuple.
    Handles format: prefix.middle-suffix where suffix can be number + optional letter.
    """
    try:
        # Split by "."
        parts = id_str.split(".")
        prefix = int(parts[0]) if parts[0].isdigit() else parts[0]
        
        # Split the rest by "-"
        rest = parts[1] if len(parts) > 1 else ""
        subparts = rest.split("-")
        
        middle = int(subparts[0]) if subparts[0].isdigit() else subparts[0]
        
        suffix_str = subparts[1] if len(subparts) > 1 else ""
        
        # Extract numeric and letter parts from suffix (e.g., "12b" -> 12, "b")
        match = re.match(r"(\d+)([a-zA-Z]*)", suffix_str)
        if match:
            suffix_num = int(match.group(1))
            suffix_letter = match.group(2) or ""
        else:
            suffix_num = 0
            suffix_letter = suffix_str
        
        return (prefix, middle, suffix_num, suffix_letter)
    except Exception:
        # Fallback: return the original string for sorting
        return (0, 0, 0, id_str)


def add_original_index(dataset):
    """
    Add _original_idx field based on proper numeric sorting of the 'id' field.
    """
    # Get all IDs with their current positions
    ids_with_pos = [(i, dataset[i]["id"]) for i in range(len(dataset))]
    
    # Sort by properly parsed ID
    ids_with_pos.sort(key=lambda x: parse_id(x[1]))
    
    # Create mapping: current dataset position -> proper sorted index
    proper_index = {orig_pos: sorted_idx for sorted_idx, (orig_pos, _) in enumerate(ids_with_pos)}

    return dataset.map(
        lambda x, idx: {**x, "_original_idx": proper_index[idx]},
        with_indices=True
    )


class PrecedingWrapSampler(ContextSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build index mapping from original index to position in df
        self._orig_to_pos = {doc["_original_idx"]: i for i, doc in enumerate(self.df)}
        # Sort docs by original index for ordered access
        self._ordered_docs = sorted(self.df, key=lambda d: d["_original_idx"])
    
    def sample(self, n: int, eval_doc=None, df=None, **kwargs):
        total = len(self._ordered_docs)
        docs = df if df is not None else self.df

        if eval_doc is None or "_original_idx" not in eval_doc:
            print(f">>> Oopsie")
            return list(self.rnd.sample(list(self.df), n))
        
        doc_id = eval_doc["_original_idx"]
        indices = [(doc_id - n + i) % total for i in range(n)]
        return [self._ordered_docs[idx] for idx in indices]