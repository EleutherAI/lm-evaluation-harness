"""Helpers splitting the Uncheatable Eval snapshot into one task per category."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import datasets


def filter_category(dataset: datasets.Dataset, category: str) -> datasets.Dataset:
    """Return the subset of the snapshot belonging to a single category."""
    return dataset.filter(lambda example: example["category"] == category)


def filter_ao3_english(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "ao3_english")


def filter_ao3_nonenglish(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "ao3_nonenglish")


def filter_arxiv_cs(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "arxiv_cs")


def filter_arxiv_math(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "arxiv_math")


def filter_arxiv_other(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "arxiv_other")


def filter_arxiv_physics(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "arxiv_physics")


def filter_bbc_news(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "bbc_news")


def filter_biorxiv_all(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "biorxiv_all")


def filter_github_cpp(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "github_cpp")


def filter_github_javascript(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "github_javascript")


def filter_github_markdown(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "github_markdown")


def filter_github_other(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "github_other")


def filter_github_python(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "github_python")


def filter_wikipedia_english(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "wikipedia_english")


def filter_wikipedia_nonenglish(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_category(dataset, "wikipedia_nonenglish")
