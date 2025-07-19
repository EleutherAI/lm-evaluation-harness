from functools import partial

CATEGORIES = ['geography', 'history', 'political_life', 'social_life', 'insurance']

def process_docs(dataset, category):
    return dataset.filter(lambda x: x["topic"] == category)


process_functions = {
    f"process_{category.lower().replace(' ', '_')}": partial(
        process_docs, category=category
    )
    for category in CATEGORIES
}

globals().update(process_functions)
