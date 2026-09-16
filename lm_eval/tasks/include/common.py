"""Shared pieces for the INCLUDE tasks.

`!function` only resolves against a `utils.py` sitting next to the yaml, so every
language directory keeps a thin `utils.py` that imports from here.
"""

import logging

from lm_eval.api.samplers import ContextSampler
from lm_eval.api.task import ConfigurableTask


eval_logger = logging.getLogger(__name__)


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

OPTION_FIELDS = ("option_a", "option_b", "option_c", "option_d")

_CANONICAL_DOMAIN = {category.casefold(): category for category in CATEGORIES}


def resolve_domain(doc):
    """Return a doc's domain, reading `subject` when `domain` is not a real domain.

    Some validation docs are filed under domain="Other" but name their domain in
    `subject`; without this fallback they have no domain-matched examples at all.
    """
    for field in ("domain", "subject"):
        value = (doc.get(field) or "").strip()
        if value.casefold() in _CANONICAL_DOMAIN:
            return _CANONICAL_DOMAIN[value.casefold()]
    return (doc.get("domain") or "").strip()


def item_key(doc):
    """Identify an item by its question and its set of options.

    Metadata columns disagree between the splits, and a shared item sometimes has its
    options reordered, so neither whole-row nor per-field comparison finds them.
    """
    options = sorted(str(doc.get(field)) for field in OPTION_FIELDS)
    return (doc.get("question"), tuple(options))


class DomainMatchedSampler(ContextSampler):
    """Draw fewshot examples from the eval doc's own domain, never the doc itself.

    The pool is the whole validation split, since its `domain` labels are unreliable.
    Per eval doc: drop the doc's own item, since the splits share some items and one
    of those would hand over the answer; prefer the doc's domain, so the task
    description stays true of the examples; backfill from other domains only if that
    leaves too few.

    Needs `IncludeTask` to supply the eval doc.
    """

    eval_doc = None

    def sample(self, n, eval_doc=None, df=None, **kwargs):
        if n == 0:
            return []
        if df:
            self.df = df
        pool = list(self.fewshot_docs())
        doc = eval_doc if eval_doc is not None else self.eval_doc
        if doc is None:
            eval_logger.warning("No eval doc; sampling without a domain preference.")
            return self.rnd.sample(pool, n)

        key = item_key(doc)
        candidates = [item for item in pool if item_key(item) != key]
        domain = resolve_domain(doc)
        matched = [item for item in candidates if resolve_domain(item) == domain]
        others = [item for item in candidates if resolve_domain(item) != domain]

        shots = self.rnd.sample(matched, min(n, len(matched)))
        if len(shots) < n:
            shots += self.rnd.sample(others, min(n - len(shots), len(others)))
        if len(shots) < n:
            eval_logger.warning(
                f"Only {len(shots)} of {n} fewshot examples available for {domain!r}."
            )
        self.rnd.shuffle(shots)
        return shots


class IncludeTask(ConfigurableTask):
    """Hand the eval doc to the fewshot sampler.

    `fewshot_context` forwards it only when the fewshot split is the test split, and
    INCLUDE draws its examples from `validation`.
    """

    def __init__(self, config=None, **kwargs):
        # `_build_task` leaves `class` in the config and `TaskConfig` rejects it.
        config = {k: v for k, v in (config or {}).items() if k != "class"}
        super().__init__(config=config, **kwargs)

    def fewshot_context(self, doc, *args, **kwargs):
        sampler = getattr(self, "sampler", None)
        if isinstance(sampler, DomainMatchedSampler):
            sampler.eval_doc = doc
        return super().fewshot_context(doc, *args, **kwargs)
