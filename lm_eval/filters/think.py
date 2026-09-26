"""Filter for reasoning-model (<think>...</think>) output.

Reasoning models emit a thinking section before their answer. Extraction
filters downstream (regex, exact-match comparisons) should see only the
answer, not the trace. This filter strips complete and unclosed thinking
blocks from each response, working at the task/filter layer so it applies
identically across every backend (hf, vllm, API).
"""

import re

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN = re.compile(r"^<think>.*", re.DOTALL)


@register_filter("strip_think")
class StripThinkFilter(Filter):
    """Remove ``<think>...</think>`` blocks (and any unclosed leading
    ``<think>`` section) from each response.

    Typical use: place before an extraction filter in a task's
    ``filter_list`` so downstream regexes match the answer, not the trace::

        filter_list:
          - name: "strip-thinking"
            filter:
              - function: "strip_think"
              - function: "regex"
                regex_pattern: "#### (-?[0-9.,]+)"
    """

    def __init__(self, think_end_token: str = "</think>") -> None:
        self.think_end_token = think_end_token

    def apply(self, resps, docs):
        def strip(resp: str) -> str:
            if not isinstance(resp, str) or self.think_end_token not in resp:
                # no closed thinking block: strip an unclosed leading one
                if isinstance(resp, str) and resp.lstrip().startswith("<think>"):
                    return _THINK_OPEN.sub("", resp.lstrip()).lstrip()
                return resp
            return resp.split(self.think_end_token, 1)[1].lstrip()

        def filter_set(inst):
            return [strip(r) for r in inst]

        return map(filter_set, resps)
