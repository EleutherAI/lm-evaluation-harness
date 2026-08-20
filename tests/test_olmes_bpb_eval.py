import json
import math
from pathlib import Path

import pytest

from scripts.olmes_bpb_eval.merge_results import build_summary
from scripts.olmes_bpb_eval.ray_driver import build_shards
from scripts.olmes_bpb_eval.run_eval import ROOT, _git, _source_snapshot_sha256


SUITE_PATH = Path(__file__).parents[1] / "scripts" / "olmes_bpb_eval" / "suite.json"
CAMPAIGN_PATH = SUITE_PATH.parent


def load_suite():
    return json.loads(SUITE_PATH.read_text(encoding="utf-8"))


def test_campaign_pins_and_preflights_minerva_math_dependencies():
    constraints = (CAMPAIGN_PATH / "constraints.txt").read_text(encoding="utf-8")
    runner = (CAMPAIGN_PATH / "run_ray.sh").read_text(encoding="utf-8")
    bootstrap = (CAMPAIGN_PATH / "bootstrap_worker_venv.py").read_text(encoding="utf-8")

    assert "antlr4-python3-runtime==4.11.0" in constraints
    assert "math-verify==0.9.0" in constraints
    assert "sympy==1.14.0" in constraints
    assert "[vllm,math]" in runner
    assert "[vllm,math]" in bootstrap
    assert "import antlr4" in bootstrap


def test_runtime_provenance_is_independent_of_worker_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert _git(["rev-parse", "--show-toplevel"]) == str(ROOT)
    assert len(_source_snapshot_sha256()) == 64


def test_suite_has_requested_22_families_plus_basic_skills():
    suite = load_suite()
    family_names = {family["name"] for family in suite["families"]}

    assert len(family_names) == 23
    assert "basic_skills" in family_names
    assert suite["excluded"] == ["mrcr", "ruler"]
    assert not family_names.intersection(suite["excluded"])
    assert {model["name"] for model in suite["models"]} == {
        "Qwen/Qwen3.5-0.8B-Base",
        "Qwen/Qwen3.5-2B-Base",
    }


def test_non_code_shards_resolve_to_unique_leaf_tasks():
    shards, family_by_task = build_shards(load_suite(), "non_code", 8)
    tasks = [task for shard in shards for task in shard["tasks"]]

    assert len(shards) == 8
    assert len(tasks) == len(set(tasks)) == len(family_by_task)
    assert all(shard["compute_bpb"] for shard in shards)
    assert not any("ruler" in task or "mrcr" in task for task in tasks)


def test_phase_can_split_gated_family_without_changing_the_suite():
    _, public_families = build_shards(
        load_suite(), "non_code", 8, exclude_families={"gpqa"}
    )
    gpqa_shards, gpqa_families = build_shards(
        load_suite(), "non_code", 8, include_families={"gpqa"}
    )

    assert set(gpqa_families.values()) == {"gpqa"}
    assert "gpqa" not in set(public_families.values())
    assert set(public_families).isdisjoint(gpqa_families)
    assert len(gpqa_shards) == 1


def test_family_filter_rejects_names_outside_the_phase():
    with pytest.raises(ValueError, match="does not belong to phase"):
        build_shards(load_suite(), "non_code", 8, include_families={"humaneval"})


def test_all_phase_never_allocates_more_than_requested_workers():
    shards, _ = build_shards(load_suite(), "all", 8)

    assert len(shards) <= 8
    assert {shard["compute_bpb"] for shard in shards} == {True, False}


def test_basic_skills_summary_uses_rc_and_mc_source_headlines():
    family = next(
        family
        for family in load_suite()["families"]
        if family["name"] == "basic_skills"
    )
    suite = {"models": [{"name": "model"}], "families": [family]}
    rows = []
    for task, acc_per_token, acc in (
        ("basic_skills_arithmetic_rc", 0.25, 0.9),
        ("basic_skills_arithmetic_mc", 0.8, 0.5),
    ):
        for metric, value in (("acc_per_token", acc_per_token), ("acc", acc)):
            rows.append(
                {
                    "Model": "model",
                    "Family": "basic_skills",
                    "Task": task,
                    "Metric": metric,
                    "Filter": "none",
                    "Value": value,
                    "Samples": 1,
                }
            )

    summary = build_summary(rows, suite)[0]

    assert summary["Original Metric"] == ("acc_per_token (_rc); acc_per_token (_mc)")
    assert summary["Original Value"] == pytest.approx(0.525)


def test_summary_uses_exact_bpb_totals_and_bbeh_adjusted_harmonic():
    suite = {
        "models": [{"name": "model"}],
        "families": [
            {
                "name": "bbeh",
                "selector": "bbeh",
                "primary_metrics": ["bbeh_acc"],
                "aggregation": "bbeh_adjusted_harmonic",
            }
        ],
    }
    rows = []
    for task, acc, ll, num_bytes, samples in (
        ("bbeh_a", 0.0, -math.log(2), 1, 1),
        ("bbeh_b", 0.5, -12 * math.log(2), 4, 2),
    ):
        for metric, value in (
            ("bbeh_acc", acc),
            ("bpb_macro", -ll / (num_bytes * math.log(2))),
            ("bpb_total_loglikelihood", ll),
            ("bpb_total_bytes", num_bytes),
        ):
            rows.append(
                {
                    "Model": "model",
                    "Family": "bbeh",
                    "Task": task,
                    "Metric": metric,
                    "Filter": "none",
                    "Value": value,
                    "Samples": samples,
                }
            )

    summary = build_summary(rows, suite)[0]

    assert summary["Original Value"] == pytest.approx(2 / (100 + 1 / 0.51))
    assert summary["BPB Macro"] == pytest.approx((1 * 1 + 3 * 2) / 3)
    assert summary["BPB Corpus"] == pytest.approx(13 / 5)


def test_multipl_e_summary_marks_bpb_unavailable():
    suite = {
        "models": [{"name": "model"}],
        "families": [
            {
                "name": "multiple",
                "selector": "multiple_pass_at_1",
                "primary_metrics": ["pass@1"],
                "aggregation": "micro",
                "compute_bpb": False,
                "bpb_status": "N/A: no canonical translated completions",
            }
        ],
    }
    rows = [
        {
            "Model": "model",
            "Family": "multiple",
            "Task": "multiple_humaneval_cpp",
            "Metric": "pass@1",
            "Filter": "create_test",
            "Value": 0.25,
            "Samples": 10,
        }
    ]

    summary = build_summary(rows, suite)[0]

    assert summary["Original Value"] == 0.25
    assert summary["BPB Macro"] is None
    assert summary["BPB Corpus"] is None
    assert summary["Status / Notes"].startswith("N/A")
