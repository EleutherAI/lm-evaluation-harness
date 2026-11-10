import json


def process_docs(dataset):
    """Parse the JSON-serialized ``metadata`` column into a Python dict.

    Every row in the HuggingFace ``PrincetonPli/LongProc`` dataset carries a
    ``metadata`` column that is a JSON string.  We deserialize it so that
    downstream Jinja templates and metric functions can access it as a dict.
    """

    def _parse(doc):
        doc["metadata_parsed"] = json.loads(doc["metadata"])
        return doc

    return dataset.map(_parse)
