"""Deterministic seeding for the IFEval checkers.

The vendored checkers in ``instructions.py`` fill missing arguments from the
``random`` module inside ``build_description`` (roughly two dozen
``random.randint`` / ``random.choice`` call sites). Drawing those from the
process-wide RNG makes a verdict depend on how many items were scored before
the current one, so the same ``(prompt, response)`` pair can score both ``True``
and ``False`` inside a single run: see issue #4602.

``deterministic_checker_draws`` runs checker construction against a ``random``
shim backed by a generator seeded from the item itself, and restores the
caller's ``random`` attribute afterwards. The drawn arguments then depend on
the item alone, and scoring no longer consumes from (or perturbs) the global
stream, so a run that seeds ``random`` for reproducibility stays reproducible.
"""

import contextlib
import hashlib
import json
import random
import threading


class _RandomShim:
    """Stand-in for the ``random`` module backed by a per-item generator."""

    def __init__(self, rng):
        self._rng = rng

    def __getattr__(self, name):
        return getattr(self._rng, name)


def stable_seed(instruction_id, prompt, kwargs):
    """Return a seed for one instruction of one item.

    ``hash`` cannot be used here because it is salted per process, so the same
    item would draw different arguments in different runs.
    """
    payload = json.dumps(
        {"instruction_id": instruction_id, "prompt": prompt, "kwargs": kwargs},
        sort_keys=True,
        default=str,
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


# Checker construction is cheap and scoring is single-threaded in the
# evaluator, but the swap below is a module-level one, so serialise it to keep
# concurrent users of these checkers from observing each other's generator.
_lock = threading.Lock()


@contextlib.contextmanager
def deterministic_checker_draws(seed, *modules):
    """Seed the ``random`` name used by ``modules`` while constructing a checker.

    ``modules`` are the instruction modules that call ``random.*`` at build
    time (for example ``lm_eval.tasks.ifeval.instructions`` and
    ``...instructions_util``). Only their module-level ``random`` attribute is
    rebound, so the global ``random`` module state is never mutated.
    """
    if not modules:
        raise ValueError("deterministic_checker_draws needs at least one module")
    shim = _RandomShim(random.Random(seed))
    saved = [(module, module.random) for module in dict.fromkeys(modules)]
    with _lock:
        for module, _ in saved:
            module.random = shim
        try:
            yield
        finally:
            for module, original in saved:
                module.random = original
