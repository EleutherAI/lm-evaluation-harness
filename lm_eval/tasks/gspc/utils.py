"""Per-bank doc filters for GSPC tasks: drop canary placeholder rows and empty items.
Banks name their content column differently (tool/item/operation/case) — each filter
matches its bank's field. Honest grading note: exact label match here is a lane metric;
canonical GSPC grading stays board_v2 at councilof.ai. Never certify."""
FIELDS = {
    "gspc_mcp": "tool",
    "gspc_oss": "item",
    "gspc_prv": "operation",
    "gspc_asi": "item",
    "gspc_det": "case",
    "gspc_art5": "item",
}


def _make_filter(field):
    def filter_ds(dataset):
        return dataset.filter(
            lambda doc: bool(str(doc.get(field) or "").strip())
            and not doc.get("_canary")
            and bool(str(doc.get("expected") or "").strip())
        )

    return filter_ds


filter_canaries_mcp = _make_filter("tool")
filter_canaries_oss = _make_filter("item")
filter_canaries_prv = _make_filter("operation")
filter_canaries_asi = _make_filter("item")
filter_canaries_det = _make_filter("case")
filter_canaries_art5 = _make_filter("item")
