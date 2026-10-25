"""Offline tests for the results table formatter in `scripts/make_table_results.py`."""

from scripts.make_table_results import make_table


def _results(entries, versions):
    return {"results": entries, "versions": versions}


def _rows(table):
    """Split a markdown table into stripped cell lists, skipping separator rows."""
    rows = []
    for line in table.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    return rows


def _row_for(table, task):
    return next(row for row in _rows(table) if row[0] == task)


def test_scales_accuracy_and_stderr_to_percent():
    table = make_table(
        _results({"arc_easy": {"acc": 0.5, "acc_stderr": 0.01}}, {"arc_easy": 0})
    )

    row = _row_for(table, "arc_easy")
    assert row[2] == "acc"
    assert row[3] == "50"
    assert row[4] == "±"
    assert row[5] == "1"


def test_keeps_perplexity_unscaled():
    table = make_table(
        _results(
            {"wikitext": {"ppl": 12.34, "ppl_stderr": 0.56}},
            {"wikitext": 1},
        )
    )

    row = _row_for(table, "wikitext")
    assert row[3] == "12.34"
    assert row[4] == "±"
    assert row[5] == "0.56"


def test_squad2_accuracy_is_not_scaled():
    table = make_table(_results({"squad2": {"acc": 0.5}}, {"squad2": 0}))

    row = _row_for(table, "squad2")
    assert row[3] == "0.5"


def test_stderr_metrics_are_skipped():
    table = make_table(
        _results({"arc_easy": {"acc": 0.5, "acc_stderr": 0.01}}, {"arc_easy": 0})
    )

    assert [row for row in _rows(table) if row[2] != "Metric"] == [
        ["arc_easy", "0", "acc", "50", "±", "1"]
    ]


def test_tasks_are_sorted_by_name():
    table = make_table(
        _results(
            {"zeta": {"acc": 0.1}, "alpha": {"acc": 0.2}},
            {"zeta": 0, "alpha": 0},
        )
    )

    tasks = [row[0] for row in _rows(table) if row[0] != "Task"]
    assert tasks == ["alpha", "zeta"]


def test_metric_without_stderr_has_no_error_cell():
    table = make_table(_results({"arc_easy": {"acc": 0.5}}, {"arc_easy": 0}))

    row = _row_for(table, "arc_easy")
    assert row[3] == "50"
    assert row[4] == ""
    assert row[5] == ""
