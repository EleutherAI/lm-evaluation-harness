"""
Utility functions for BLEnD dataset tasks.
BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages
"""

import json
from functools import partial


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

    filtered_dataset = dataset.filter(lambda x: x["country"] == country)
    return filtered_dataset.map(parse_choices)


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
