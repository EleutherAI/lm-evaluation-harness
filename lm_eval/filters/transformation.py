from lm_eval.api.filter import Filter


class LowercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.lower() for resp in inst]

        return [filter_set(resp) for resp in resps]


class UppercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.upper() for resp in inst]

        return [filter_set(resp) for resp in resps]


class MapFilter(Filter):
    def __init__(self, mapping_dict: dict = {}, default_value=None) -> None:
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
        assert isinstance(
            mapping_dict, dict
        ), "Provided mapping_dict is not a dictionary"
        self.mapping_dict = mapping_dict
        self.default_value = default_value

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp, self.default_value) for resp in inst]

        return [filter_set(resp) for resp in resps]
