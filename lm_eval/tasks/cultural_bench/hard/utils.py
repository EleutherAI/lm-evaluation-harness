"""
Utility functions for CulturalBench dataset tasks.
CulturalBench: A Benchmark for Evaluating Cultural Awareness in Large Language Models
"""


def process_docs_by_country(dataset, country):
    """
    Filter dataset by country and convert answers to appropriate format.

    Args:
        dataset: The dataset to filter
        country: The country name to filter by

    Returns:
        Filtered dataset containing only questions for the specified country
    """

    def convert_answers(doc):
        """Convert answer format based on dataset type"""
        # For easy (multiple choice): convert letter to index
        if doc["answer"] in ["A", "B", "C", "D"]:
            doc["answer"] = "ABCD".index(doc["answer"])
        # For hard (true/false): convert boolean to index
        elif isinstance(doc["answer"], bool):
            doc["answer"] = 1 if doc["answer"] else 0
        return doc

    filtered_dataset = dataset.filter(lambda x: x["country"] == country)
    return filtered_dataset.map(convert_answers)


# Create individual process functions for specific countries
def process_argentina(dataset):
    return process_docs_by_country(dataset, "Argentina")


def process_australia(dataset):
    return process_docs_by_country(dataset, "Australia")


def process_bangladesh(dataset):
    return process_docs_by_country(dataset, "Bangladesh")


def process_brazil(dataset):
    return process_docs_by_country(dataset, "Brazil")


def process_canada(dataset):
    return process_docs_by_country(dataset, "Canada")


def process_chile(dataset):
    return process_docs_by_country(dataset, "Chile")


def process_china(dataset):
    return process_docs_by_country(dataset, "China")


def process_czech_republic(dataset):
    return process_docs_by_country(dataset, "Czech Republic")


def process_egypt(dataset):
    return process_docs_by_country(dataset, "Egypt")


def process_france(dataset):
    return process_docs_by_country(dataset, "France")


def process_germany(dataset):
    return process_docs_by_country(dataset, "Germany")


def process_hong_kong(dataset):
    return process_docs_by_country(dataset, "Hong Kong")


def process_india(dataset):
    return process_docs_by_country(dataset, "India")


def process_indonesia(dataset):
    return process_docs_by_country(dataset, "Indonesia")


def process_iran(dataset):
    return process_docs_by_country(dataset, "Iran")


def process_israel(dataset):
    return process_docs_by_country(dataset, "Israel")


def process_italy(dataset):
    return process_docs_by_country(dataset, "Italy")


def process_japan(dataset):
    return process_docs_by_country(dataset, "Japan")


def process_lebanon(dataset):
    return process_docs_by_country(dataset, "Lebanon")


def process_malaysia(dataset):
    return process_docs_by_country(dataset, "Malaysia")


def process_mexico(dataset):
    return process_docs_by_country(dataset, "Mexico")


def process_morocco(dataset):
    return process_docs_by_country(dataset, "Morocco")


def process_nepal(dataset):
    return process_docs_by_country(dataset, "Nepal")


def process_netherlands(dataset):
    return process_docs_by_country(dataset, "Netherlands")


def process_new_zealand(dataset):
    return process_docs_by_country(dataset, "New Zealand")


def process_nigeria(dataset):
    return process_docs_by_country(dataset, "Nigeria")


def process_pakistan(dataset):
    return process_docs_by_country(dataset, "Pakistan")


def process_peru(dataset):
    return process_docs_by_country(dataset, "Peru")


def process_philippines(dataset):
    return process_docs_by_country(dataset, "Philippines")


def process_poland(dataset):
    return process_docs_by_country(dataset, "Poland")


def process_romania(dataset):
    return process_docs_by_country(dataset, "Romania")


def process_russia(dataset):
    return process_docs_by_country(dataset, "Russia")


def process_saudi_arabia(dataset):
    return process_docs_by_country(dataset, "Saudi Arabia")


def process_singapore(dataset):
    return process_docs_by_country(dataset, "Singapore")


def process_south_africa(dataset):
    return process_docs_by_country(dataset, "South Africa")


def process_south_korea(dataset):
    return process_docs_by_country(dataset, "South Korea")


def process_spain(dataset):
    return process_docs_by_country(dataset, "Spain")


def process_taiwan(dataset):
    return process_docs_by_country(dataset, "Taiwan")


def process_thailand(dataset):
    return process_docs_by_country(dataset, "Thailand")


def process_turkey(dataset):
    return process_docs_by_country(dataset, "Turkey")


def process_ukraine(dataset):
    return process_docs_by_country(dataset, "Ukraine")


def process_united_kingdom(dataset):
    return process_docs_by_country(dataset, "United Kingdom")


def process_united_states(dataset):
    return process_docs_by_country(dataset, "United States")


def process_vietnam(dataset):
    return process_docs_by_country(dataset, "Vietnam")


def process_zimbabwe(dataset):
    return process_docs_by_country(dataset, "Zimbabwe")
