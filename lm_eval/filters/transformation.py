import re

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


@register_filter("lowercase")
class LowercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.lower() for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("uppercase")
class UppercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.upper() for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("map")
class MapFilter(Filter):
    def __init__(self, mapping_dict: dict = None, default_value=None) -> None:
        """
        Initializes the MapFilter with a given mapping dictionary and default value.

        Args:
        - mapping_dict (dict): A dictionary containing the key-value mappings.
                               Default is an empty dictionary.
        - default_value (Any): The value to be returned when a key is not found in the mapping_dict.
                               Default is None.

        Example:
        mapper = MapFilter({'A': 1, 'B': 2}, default_value=0)
        """
        if mapping_dict is None:
            mapping_dict = {}
        assert isinstance(mapping_dict, dict), (
            "Provided mapping_dict is not a dictionary"
        )
        self.mapping_dict = mapping_dict
        self.default_value = default_value

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp, self.default_value) for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("format_span")
class SPANFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def format_ner_text(text):
            label_dict = {
                "person": "PER",
                "location": "LOC",
                "organization": "ORG",
                "counties": "LOC",
                "places": "LOC",
                "people": "PER",
                "persons": "PER",
                "company": "ORG",
                "country": "LOC",
                "continent": "LOC",
                "time": "DATE",
                "date": "DATE",
                "per": "PER",
                "loc": "LOC",
                "org": "ORG",
            }
            text = text.lower()
            for key, value in label_dict.items():
                text = text.replace(key, value)

            text = "$".join(i for i in text.split("$$"))
            return text.rstrip("$$")

        def format_named_entities(text):
            """
            Extract named entities from text and format them as 'label: value $$ label: value'.
            Handles grouped entities (e.g., LOC: kenya, uganda) and excludes 'none' values.
            """
            # Regular expression to match label: entities pattern
            pattern = r"\b(PER|LOC|ORG|DATE):\s*([^$]+)"
            # Normalize newline characters
            text = text.replace("\n", "$").strip()
            matches = re.findall(pattern, text)

            formatted_entities = []

            for label, values in matches:
                # Split multiple entities separated by commas and strip whitespace
                entities = [value.strip() for value in values.split(",")]

                # Exclude 'none' entities
                for entity in entities:
                    if entity.lower() != "none":
                        formatted_entities.append(f"{label.lower()}: {entity}")

            # Join entities with the desired separator
            return " $ ".join(formatted_entities)

        def filter_set(inst):
            return [
                format_named_entities(format_ner_text(resp.lower())) for resp in inst
            ]

        return [filter_set(resp) for resp in resps]
