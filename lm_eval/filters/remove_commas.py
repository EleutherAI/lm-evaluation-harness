from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


@register_filter("remove_commas")
class RemoveCommasFilter(Filter):
    """A filter that removes commas from strings.
    
    Useful for normalizing numbers extracted from text (e.g., converting "1,234" to "1234").
    """
    
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.replace(",", "") for resp in inst]

        return [filter_set(resp) for resp in resps]


@register_filter("clean_number")
class CleanNumberFilter(Filter):
    """A filter that cleans extracted numbers by removing commas, spaces, newlines, dollar signs, and 'x'.
    
    Useful for normalizing numbers extracted from GSM8K answers.
    """
    
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def clean_text(text):
            """Clean a single text string"""
            return (text
                .replace(",", "")
                .replace(" ", "")
                .replace("\n", "")
                .replace("$", "")
                .replace("x", ""))
        
        def filter_set(inst):
            # Handle both single strings and lists
            if isinstance(inst, str):
                return clean_text(inst)
            elif isinstance(inst, list):
                return [clean_text(resp) for resp in inst]
            else:
                return inst
        
        # Apply to all responses
        return [filter_set(resp) for resp in resps]