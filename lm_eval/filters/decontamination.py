from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


@register_filter("decontaminate")
class DecontaminationFilter(Filter):
    """
    A filter which evaluates
    """

    name = "track_decontamination"

    def __init__(self, path) -> None:
        """

        TODO: make sure only ever run one time on the train set (should this be cached as a class var? keyed by value for "path").
        should further cache result on a given (task_name, doc_id)
        """
        self._decontam_results = None

    def apply(self, resps, docs) -> None:
        """
        Return {"no_contamination", "only_contamination"} keys for the 2 different subsets
        """
        pass
