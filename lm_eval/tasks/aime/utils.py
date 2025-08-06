import re
from datasets import Dataset


def extract_year_from_url(url: str) -> str:
    """Extract the year from an AIME problem URL."""
    match = re.search(r"index\.php/(\d{4})_", url)
    if not match:
        raise ValueError(f"Could not extract year from URL: {{url}}")
    return match.group(1)

def process_2022(dataset: Dataset) -> Dataset:
    """Filter dataset to only include problems from 2022."""

    def filter_by_year(doc):
        try:
            doc_year = extract_year_from_url(doc["url"])
            return doc_year == "2022"
        except ValueError:
            return False

    return dataset.filter(filter_by_year)

def process_2023(dataset: Dataset) -> Dataset:
    """Filter dataset to only include problems from 2023."""

    def filter_by_year(doc):
        try:
            doc_year = extract_year_from_url(doc["url"])
            return doc_year == "2023"
        except ValueError:
            return False

    return dataset.filter(filter_by_year)

def process_2024(dataset: Dataset) -> Dataset:
    """Filter dataset to only include problems from 2024."""

    def filter_by_year(doc):
        try:
            doc_year = extract_year_from_url(doc["url"])
            return doc_year == "2024"
        except ValueError:
            return False

    return dataset.filter(filter_by_year)

