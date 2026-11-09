from typing import List

import numpy as np


try:
    import tinyBenchmarks as tb
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`tinyBenchmarks` is required for tinyBenchmarks task metric calculation, install via \
`pip install git+https://github.com/felipemaiapolo/tinyBenchmarks`"
    )


def agg_pirt(items: List[float], benchmark: str) -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["pirt"]


def agg_gpirt_arc(items: List[float], benchmark: str = "arc") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_gsm8k(items: List[float], benchmark: str = "gsm8k") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_hellaswag(items: List[float], benchmark: str = "hellaswag") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_mmlu(items: List[float], benchmark: str = "mmlu") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_truthfulqa(items: List[float], benchmark: str = "truthfulqa") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_winogrande(items: List[float], benchmark: str = "winogrande") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]
