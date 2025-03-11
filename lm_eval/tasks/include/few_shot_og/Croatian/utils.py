from functools import partial


CATEGORIES = [
    "Applied Science",
    "Arts & Humanities",
    "Business & Commerce",
    "Driving License",
    "General knowledge",
    "Health oriented education",
    "Marine License",
    "Medical License",
    "Professional certification",
    "STEM",
    "Social Science",
]


def process_docs(dataset, category):
    return dataset.filter(lambda x: x["domain"] == category)


process_functions = {
    f"process_{category.lower().replace(' & ', '_').replace(' ', '_')}": partial(
        process_docs, category=category
    )
    for category in CATEGORIES
}

globals().update(process_functions)
