import numpy as np


try:
    import tinyBenchmarks as tb
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`tinyBenchmarks` is required for tinyBenchmarks task metric calculation, install via \
`pip install git+https://github.com/felipemaiapolo/tinyBenchmarks`"
    )


def agg_pirt(items: list[float], benchmark: str) -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["pirt"]


def agg_gpirt_arc(items: list[float], benchmark: str = "arc") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_gsm8k(items: list[float], benchmark: str = "gsm8k") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_hellaswag(items: list[float], benchmark: str = "hellaswag") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_mmlu(items: list[float], benchmark: str = "mmlu") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_truthfulqa(items: list[float], benchmark: str = "truthfulqa") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]


def agg_gpirt_winogrande(items: list[float], benchmark: str = "winogrande") -> float:
    items = np.array(items)
    predictions = tb.evaluate(items, benchmark)
    return predictions[benchmark]["gpirt"]
