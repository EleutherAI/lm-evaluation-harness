from functools import partial


CATEGORIES = ["Business", "Humanities", "Medical", "Other", "STEM", "Social Sciences"]


def process_docs(dataset, category):
    return dataset.filter(lambda x: x["subject_category"] == category)


process_functions = {
    f"process_{category.lower().replace(' ', '_')}": partial(
        process_docs, category=category
    )
    for category in CATEGORIES
}

globals().update(process_functions)
