"""
Utility functions for BLEnD dataset tasks.
BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages
"""

import json
from functools import partial
from datasets import concatenate_datasets


def balanced_language_category_sample(ds, k, seed=42):
    """
    Sample k examples per country, evenly distributed across categories.

    Args:
        ds (datasets.Dataset): The dataset with columns "country" and "ID".
        k (int): Number of samples to take per language.
        seed (int): Random seed for reproducibility.

    Returns:
        datasets.Dataset: The balanced sampled dataset.
    """
    all_lang_samples = []

    for lang in ds.unique("country"):
        # Subset to this language
        ds_lang = ds.filter(lambda ex, l=lang: ex["country"] == l)

        cats = ds_lang.unique("ID")
        num_cats = len(cats)
        if num_cats == 0:
            continue

        # Even target per category for this language
        base = k // num_cats
        rem = k % num_cats
        cats_sorted = sorted(cats)
        targets = {c: base + (i < rem) for i, c in enumerate(cats_sorted)}

        per_cat_samples = []
        leftovers = []

        # Sample from each category
        for c in cats_sorted:
            ds_cat = ds_lang.filter(lambda ex, cat=c: ex["ID"] == cat).shuffle(seed=seed)
            want = targets[c]
            take = min(want, len(ds_cat))
            if take > 0:
                per_cat_samples.append(ds_cat.select(range(take)))
            if len(ds_cat) > take:
                leftovers.append(ds_cat.select(range(take, len(ds_cat))))

        # Merge category samples for this language
        lang_sample = concatenate_datasets(per_cat_samples) if per_cat_samples else None

        # Top up if some categories were too small
        got = len(lang_sample) if lang_sample is not None else 0
        need = max(0, k - got)
        if need > 0 and leftovers:
            leftover_pool = concatenate_datasets(leftovers).shuffle(seed=seed)
            top_up = leftover_pool.select(range(min(need, len(leftover_pool))))
            lang_sample = top_up if lang_sample is None else concatenate_datasets([lang_sample, top_up])

        if lang_sample is not None:
            lang_sample = lang_sample.shuffle(seed=seed)
            all_lang_samples.append(lang_sample)

    # Combine all languages
    final_sample = concatenate_datasets(all_lang_samples).shuffle(seed=seed)
    return final_sample


def process_docs_by_country(dataset, country):
    """
    Filter dataset by country and parse JSON choices.

    Args:
        dataset: The dataset to filter
        country: The country name to filter by

    Returns:
        Filtered dataset containing only questions for the specified country
    """

    def parse_choices(doc):
        """Parse JSON choices and add individual choice fields"""
        choices_dict = json.loads(doc["choices"])
        doc["choice_A"] = choices_dict["A"]
        doc["choice_B"] = choices_dict["B"]
        doc["choice_C"] = choices_dict["C"]
        doc["choice_D"] = choices_dict["D"]

        # Clean the prompt to remove JSON format instruction
        doc["clean_prompt"] = doc["prompt"].split("Provide as JSON format")[0].strip()

        return doc

    filtered_dataset = dataset.filter(lambda x: x["country"] == country).select(range(500))
    sampled_dataset = balanced_language_category_sample(filtered_dataset, 500)
    return sampled_dataset.map(parse_choices)


# Create process functions for specific countries
process_algeria = partial(process_docs_by_country, country="Algeria")
process_assam = partial(process_docs_by_country, country="Assam")
process_azerbaijan = partial(process_docs_by_country, country="Azerbaijan")
process_china = partial(process_docs_by_country, country="China")
process_ethiopia = partial(process_docs_by_country, country="Ethiopia")
process_greece = partial(process_docs_by_country, country="Greece")
process_indonesia = partial(process_docs_by_country, country="Indonesia")
process_iran = partial(process_docs_by_country, country="Iran")
process_mexico = partial(process_docs_by_country, country="Mexico")
process_north_korea = partial(process_docs_by_country, country="North_Korea")
process_northern_nigeria = partial(process_docs_by_country, country="Northern_Nigeria")
process_south_korea = partial(process_docs_by_country, country="South_Korea")
process_spain = partial(process_docs_by_country, country="Spain")
process_uk = partial(process_docs_by_country, country="UK")
process_us = partial(process_docs_by_country, country="US")
process_west_java = partial(process_docs_by_country, country="West_Java")
