#!/usr/bin/env python3
"""
Script to generate CulturalBench task files for all countries.
"""

import os
from pathlib import Path

# List of all countries in CulturalBench dataset
COUNTRIES = [
    "Argentina",
    "Australia",
    "Bangladesh",
    "Brazil",
    "Canada",
    "Chile",
    "China",
    "Czech Republic",
    "Egypt",
    "France",
    "Germany",
    "Hong Kong",
    "India",
    "Indonesia",
    "Iran",
    "Israel",
    "Italy",
    "Japan",
    "Lebanon",
    "Malaysia",
    "Mexico",
    "Morocco",
    "Nepal",
    "Netherlands",
    "New Zealand",
    "Nigeria",
    "Pakistan",
    "Peru",
    "Philippines",
    "Poland",
    "Romania",
    "Russia",
    "Saudi Arabia",
    "Singapore",
    "South Africa",
    "South Korea",
    "Spain",
    "Taiwan",
    "Thailand",
    "Turkey",
    "Ukraine",
    "United Kingdom",
    "United States",
    "Vietnam",
    "Zimbabwe",
]


def country_to_filename(country):
    """Convert country name to valid filename"""
    return country.lower().replace(" ", "_")


def country_to_function_name(country):
    """Convert country name to valid Python function name"""
    return country.lower().replace(" ", "_").replace("-", "_")


def create_easy_task(country):
    """Create easy task YAML content for a country"""
    return f"""include: _easy_template_yaml
task: cultural_bench_easy_{country_to_filename(country)}
process_docs: !function utils.process_{country_to_function_name(country)}
"""


def create_hard_task(country):
    """Create hard task YAML content for a country"""
    return f"""include: _hard_template_yaml
task: cultural_bench_hard_{country_to_filename(country)}
process_docs: !function utils.process_{country_to_function_name(country)}
"""


def main():
    """Generate all country-specific task files"""

    # Create directories
    easy_dir = Path("lm_eval/tasks/cultural_bench/easy")
    hard_dir = Path("lm_eval/tasks/cultural_bench/hard")

    easy_dir.mkdir(parents=True, exist_ok=True)
    hard_dir.mkdir(parents=True, exist_ok=True)

    # Generate easy tasks
    for country in COUNTRIES:
        filename = f"cultural_bench_easy_{country_to_filename(country)}.yaml"
        filepath = easy_dir / filename

        with open(filepath, "w") as f:
            f.write(create_easy_task(country))

        print(f"Created: {filepath}")

    # Generate hard tasks
    for country in COUNTRIES:
        filename = f"cultural_bench_hard_{country_to_filename(country)}.yaml"
        filepath = hard_dir / filename

        with open(filepath, "w") as f:
            f.write(create_hard_task(country))

        print(f"Created: {filepath}")

    # Create overall group tasks
    easy_countries = [
        f"cultural_bench_easy_{country_to_filename(country)}" for country in COUNTRIES
    ]
    hard_countries = [
        f"cultural_bench_hard_{country_to_filename(country)}" for country in COUNTRIES
    ]
    all_tasks = easy_countries + hard_countries

    # Easy group task
    easy_group_content = f"""group: cultural_bench_easy
task:
{chr(10).join(f'  - {task}' for task in easy_countries)}
"""

    with open("lm_eval/tasks/cultural_bench/cultural_bench_easy.yaml", "w") as f:
        f.write(easy_group_content)
    print("Created: lm_eval/tasks/cultural_bench/cultural_bench_easy.yaml")

    # Hard group task
    hard_group_content = f"""group: cultural_bench_hard
task:
{chr(10).join(f'  - {task}' for task in hard_countries)}
"""

    with open("lm_eval/tasks/cultural_bench/cultural_bench_hard.yaml", "w") as f:
        f.write(hard_group_content)
    print("Created: lm_eval/tasks/cultural_bench/cultural_bench_hard.yaml")

    # Overall group task
    overall_group_content = f"""group: cultural_bench
task:
  - cultural_bench_easy
  - cultural_bench_hard
"""

    with open("lm_eval/tasks/cultural_bench/cultural_bench.yaml", "w") as f:
        f.write(overall_group_content)
    print("Created: lm_eval/tasks/cultural_bench/cultural_bench.yaml")

    print(f"\nGenerated {len(COUNTRIES) * 2} country-specific tasks + 3 group tasks")
    print(f"Countries: {len(COUNTRIES)}")


if __name__ == "__main__":
    main()
