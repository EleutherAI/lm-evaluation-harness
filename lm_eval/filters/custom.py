from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


@register_filter("custom")
class CustomFilter(Filter):
    """
    Custom filter that applies a custom, user-defined function to the model responses.
    """

    def __init__(self, **kwargs) -> None:
        self.filter_fn = kwargs.pop("filter_fn")

        super().__init__(**kwargs)

    def apply(self, resps, docs):
        return self.filter_fn(resps, docs)
