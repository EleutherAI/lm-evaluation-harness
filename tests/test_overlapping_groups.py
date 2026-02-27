"""End-to-end tests for overlapping groups (shared tasks across groups).

Uses DummyLM (CPU / gloo-compatible) so no real model weights are needed.
Exercises the full pipeline: TaskManager.load() -> evaluate() ->
_process_results() -> group aggregation -> EvalResults.

Expected accuracy with DummyLM + custom datasets
-------------------------------------------------
DummyLM.loglikelihood returns (-float(doc_id), doc_id % 2 == 0) for every
request — all choices of a document get the same loglikelihood, so
np.argmax always picks index 0.

* custom_df_1: targets alternate 0, 1, 0, 1, ...  →  acc = 0.5
* custom_df_2: targets are always 0               →  acc = 1.0
* custom_df_3: test split alternates (acc = 0.5), validation split all 0 (acc = 1.0)

Both datasets use equal-length choices ("choice1", "choice2"), so
acc_norm == acc (character-length normalization has no effect).

Group-level weighted aggregation (equal sample sizes):
  overlap_group_a (df_1 + df_2) → (0.5 * N + 1.0 * N) / 2N = 0.75
  overlap_group_b (df_1 only)   → 0.5

Overlapping task with config overrides (TestConfigOverrideScoring):
  overlap_group_c: df_3 on test split (acc=0.5) + df_1 (acc=0.5)  → group = 0.5
  overlap_group_d: df_3 on val  split (acc=1.0) + df_1 (acc=0.5)  → group = 0.75
  Same task_with_custom_df_3, different test_split overrides → different scores.
"""

import os
from pathlib import Path

import pytest

from lm_eval.evaluator import evaluate
from lm_eval.models.dummy import DummyLM
from lm_eval.tasks import TaskManager


os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEST_CONFIGS = str(Path(__file__).parent / "test_configs")
LIMIT = 10  # docs per task – keeps tests fast while still meaningful

# --- Expected values (see module docstring for derivation) ---------------

EXPECTED_ACC_DF1 = 0.5  # 5 of 10 even doc_ids correct
EXPECTED_ACC_DF2 = 1.0  # all targets are 0, DummyLM always picks 0

# Choices are same length → normalization is a no-op
EXPECTED_ACC_NORM_DF1 = EXPECTED_ACC_DF1
EXPECTED_ACC_NORM_DF2 = EXPECTED_ACC_DF2

# Group A weighted mean: (0.5 * 10 + 1.0 * 10) / 20
EXPECTED_GROUP_A_ACC = 0.75
# Group B weighted mean: just the one task
EXPECTED_GROUP_B_ACC = 0.5


@pytest.fixture(scope="module")
def lm():
    return DummyLM()


@pytest.fixture(scope="module")
def tm():
    return TaskManager(include_path=TEST_CONFIGS)


@pytest.fixture(scope="module")
def results(lm, tm):
    """Run the full pipeline once; all tests in this module share the results."""
    loaded = tm.load(["overlap_group_a", "overlap_group_b"])
    return evaluate(
        lm=lm,
        task_dict=loaded,
        limit=LIMIT,
        bootstrap_iters=0,
    )


# ── Structural tests ────────────────────────────────────────────────────


class TestStructure:
    """Verify the result dict has the right keys and shape."""

    def test_namespaced_task_keys(self, results):
        """Shared task appears under each group's namespace."""
        r = results["results"]
        assert "overlap_group_a::task_with_custom_df_1" in r
        assert "overlap_group_b::task_with_custom_df_1" in r
        # Non-shared task only under group A
        assert "overlap_group_a::task_with_custom_df_2" in r

    def test_group_keys_in_results(self, results):
        """Both groups appear in results."""
        r = results["results"]
        assert "overlap_group_a" in r
        assert "overlap_group_b" in r

    def test_groups_section_present(self, results):
        """Groups with aggregate_metric_list appear in the 'groups' section."""
        assert "groups" in results
        assert "overlap_group_a" in results["groups"]
        assert "overlap_group_b" in results["groups"]

    def test_group_subtasks_mapping(self, results):
        subs = results["group_subtasks"]
        assert "overlap_group_a::task_with_custom_df_1" in subs["overlap_group_a"]
        assert "overlap_group_a::task_with_custom_df_2" in subs["overlap_group_a"]
        assert "overlap_group_b::task_with_custom_df_1" in subs["overlap_group_b"]
        assert len(subs["overlap_group_b"]) == 1

    def test_alias_is_bare_task_name(self, results):
        """Display alias strips the group namespace prefix."""
        for key in (
            "overlap_group_a::task_with_custom_df_1",
            "overlap_group_b::task_with_custom_df_1",
        ):
            alias = results["results"][key].get("alias")
            assert alias == "task_with_custom_df_1", (
                f"Expected bare name for {key}, got {alias!r}"
            )

    def test_group_aliases(self, results):
        assert results["results"]["overlap_group_a"]["alias"] == "Overlap A"
        assert results["results"]["overlap_group_b"]["alias"] == "Overlap B"

    def test_versions(self, results):
        v = results["versions"]
        for key in (
            "overlap_group_a::task_with_custom_df_1",
            "overlap_group_a::task_with_custom_df_2",
            "overlap_group_b::task_with_custom_df_1",
        ):
            assert v[key] == 1.0

    def test_n_samples(self, results):
        for key in (
            "overlap_group_a::task_with_custom_df_1",
            "overlap_group_a::task_with_custom_df_2",
            "overlap_group_b::task_with_custom_df_1",
        ):
            ns = results["n-samples"][key]
            assert ns["original"] == 100, f"Expected 100 total docs for {key}"
            assert ns["effective"] == LIMIT, (
                f"Expected {LIMIT} effective docs for {key}"
            )


# ── Task-level scoring tests ────────────────────────────────────────────


class TestTaskScoring:
    """Verify per-task accuracy values are exactly as predicted."""

    def test_custom_df_1_accuracy_in_group_a(self, results):
        m = results["results"]["overlap_group_a::task_with_custom_df_1"]
        assert m["acc,none"] == pytest.approx(EXPECTED_ACC_DF1)

    def test_custom_df_1_accuracy_in_group_b(self, results):
        m = results["results"]["overlap_group_b::task_with_custom_df_1"]
        assert m["acc,none"] == pytest.approx(EXPECTED_ACC_DF1)

    def test_custom_df_2_accuracy(self, results):
        m = results["results"]["overlap_group_a::task_with_custom_df_2"]
        assert m["acc,none"] == pytest.approx(EXPECTED_ACC_DF2)

    def test_custom_df_1_acc_norm_in_group_a(self, results):
        m = results["results"]["overlap_group_a::task_with_custom_df_1"]
        assert m["acc_norm,none"] == pytest.approx(EXPECTED_ACC_NORM_DF1)

    def test_custom_df_1_acc_norm_in_group_b(self, results):
        m = results["results"]["overlap_group_b::task_with_custom_df_1"]
        assert m["acc_norm,none"] == pytest.approx(EXPECTED_ACC_NORM_DF1)

    def test_custom_df_2_acc_norm(self, results):
        m = results["results"]["overlap_group_a::task_with_custom_df_2"]
        assert m["acc_norm,none"] == pytest.approx(EXPECTED_ACC_NORM_DF2)

    def test_shared_task_scores_are_identical(self, results):
        """The same task in two groups must produce the same metrics."""
        a = results["results"]["overlap_group_a::task_with_custom_df_1"]
        b = results["results"]["overlap_group_b::task_with_custom_df_1"]
        assert a["acc,none"] == pytest.approx(b["acc,none"])
        assert a["acc_norm,none"] == pytest.approx(b["acc_norm,none"])
        assert a["acc_stderr,none"] == pytest.approx(b["acc_stderr,none"])
        assert a["acc_norm_stderr,none"] == pytest.approx(b["acc_norm_stderr,none"])
        assert a["sample_len"] == b["sample_len"]

    def test_sample_len(self, results):
        for key in (
            "overlap_group_a::task_with_custom_df_1",
            "overlap_group_a::task_with_custom_df_2",
            "overlap_group_b::task_with_custom_df_1",
        ):
            assert results["results"][key]["sample_len"] == LIMIT

    def test_stderr_is_na_without_bootstrap(self, results):
        """With bootstrap_iters=0, stderr is reported as 'N/A'."""
        for key in (
            "overlap_group_a::task_with_custom_df_1",
            "overlap_group_a::task_with_custom_df_2",
            "overlap_group_b::task_with_custom_df_1",
        ):
            m = results["results"][key]
            assert "acc_stderr,none" in m
            assert m["acc_stderr,none"] == "N/A"


# ── Group-level aggregation tests ───────────────────────────────────────


class TestGroupAggregation:
    """Verify that group-level metrics are correct weighted averages."""

    def test_group_a_acc(self, results):
        """Group A = weighted mean of df_1 (0.5) and df_2 (1.0) → 0.75."""
        g = results["groups"]["overlap_group_a"]
        assert g["acc,none"] == pytest.approx(EXPECTED_GROUP_A_ACC)

    def test_group_b_acc(self, results):
        """Group B has only df_1 → 0.5."""
        g = results["groups"]["overlap_group_b"]
        assert g["acc,none"] == pytest.approx(EXPECTED_GROUP_B_ACC)

    def test_group_a_acc_norm(self, results):
        g = results["groups"]["overlap_group_a"]
        assert g["acc_norm,none"] == pytest.approx(EXPECTED_GROUP_A_ACC)

    def test_group_b_acc_norm(self, results):
        g = results["groups"]["overlap_group_b"]
        assert g["acc_norm,none"] == pytest.approx(EXPECTED_GROUP_B_ACC)

    def test_group_a_sample_len(self, results):
        """Group A aggregates two tasks → double the per-task limit."""
        g = results["groups"]["overlap_group_a"]
        assert g["sample_len"] == 2 * LIMIT

    def test_group_b_sample_len(self, results):
        g = results["groups"]["overlap_group_b"]
        assert g["sample_len"] == LIMIT

    def test_groups_have_different_acc(self, results):
        """The whole point of overlapping groups: same shared task, different group-level results because of different compositions."""
        ga = results["groups"]["overlap_group_a"]["acc,none"]
        gb = results["groups"]["overlap_group_b"]["acc,none"]
        assert ga != pytest.approx(gb)

    def test_group_stderr_present(self, results):
        """Group stderr keys should be present."""
        for grp in ("overlap_group_a", "overlap_group_b"):
            g = results["groups"][grp]
            assert "acc_stderr,none" in g


# ── Higher-is-better metadata ───────────────────────────────────────────


class TestHigherIsBetter:
    def test_task_higher_is_better(self, results):
        hib = results["higher_is_better"]
        for key in (
            "overlap_group_a::task_with_custom_df_1",
            "overlap_group_a::task_with_custom_df_2",
            "overlap_group_b::task_with_custom_df_1",
        ):
            assert hib[key]["acc"] is True
            assert hib[key]["acc_norm"] is True

    def test_group_higher_is_better(self, results):
        hib = results["higher_is_better"]
        for grp in ("overlap_group_a", "overlap_group_b"):
            assert hib[grp]["acc"] is True
            assert hib[grp]["acc_norm"] is True


# ── Config-override scoring tests ───────────────────────────────────────
#
# overlap_group_c: task_with_custom_df_3 (default test split → acc=0.5) + df_1
# overlap_group_d: task_with_custom_df_3 (test_split: validation → acc=1.0) + df_1
#
# Same task name, different split overrides → different per-task scores.


EXPECTED_DF3_TEST_ACC = 0.5  # test split alternates targets
EXPECTED_DF3_VAL_ACC = 1.0  # validation split has all-zero targets

# Group C: (0.5 + 0.5) / 2 = 0.5  (both tasks at 0.5)
EXPECTED_GROUP_C_ACC = 0.5
# Group D: (1.0 + 0.5) / 2 = 0.75  (df_3 at 1.0, df_1 at 0.5)
EXPECTED_GROUP_D_ACC = 0.75


@pytest.fixture(scope="module")
def override_results(lm, tm):
    """Run the pipeline with groups C and D that override test_split."""
    loaded = tm.load(["overlap_group_c", "overlap_group_d"])
    return evaluate(
        lm=lm,
        task_dict=loaded,
        limit=LIMIT,
        bootstrap_iters=0,
    )


class TestConfigOverrideScoring:
    """Same task in two groups with different config overrides → different scores."""

    def test_shared_task_has_different_acc(self, override_results):
        """df_3 on test split (0.5) vs validation split (1.0)."""
        c = override_results["results"]["overlap_group_c::task_with_custom_df_3"]
        d = override_results["results"]["overlap_group_d::task_with_custom_df_3"]
        assert c["acc,none"] == pytest.approx(EXPECTED_DF3_TEST_ACC)
        assert d["acc,none"] == pytest.approx(EXPECTED_DF3_VAL_ACC)
        assert c["acc,none"] != pytest.approx(d["acc,none"])

    def test_shared_task_has_different_acc_norm(self, override_results):
        c = override_results["results"]["overlap_group_c::task_with_custom_df_3"]
        d = override_results["results"]["overlap_group_d::task_with_custom_df_3"]
        assert c["acc_norm,none"] == pytest.approx(EXPECTED_DF3_TEST_ACC)
        assert d["acc_norm,none"] == pytest.approx(EXPECTED_DF3_VAL_ACC)

    def test_unmodified_task_same_in_both(self, override_results):
        """df_1 has no override — should be identical in both groups."""
        c = override_results["results"]["overlap_group_c::task_with_custom_df_1"]
        d = override_results["results"]["overlap_group_d::task_with_custom_df_1"]
        assert c["acc,none"] == pytest.approx(d["acc,none"])
        assert c["acc,none"] == pytest.approx(EXPECTED_ACC_DF1)

    def test_group_c_aggregation(self, override_results):
        """Both tasks at 0.5 → group acc = 0.5."""
        g = override_results["groups"]["overlap_group_c"]
        assert g["acc,none"] == pytest.approx(EXPECTED_GROUP_C_ACC)

    def test_group_d_aggregation(self, override_results):
        """df_3 at 1.0 + df_1 at 0.5 → group acc = 0.75."""
        g = override_results["groups"]["overlap_group_d"]
        assert g["acc,none"] == pytest.approx(EXPECTED_GROUP_D_ACC)

    def test_groups_differ_due_to_override(self, override_results):
        """Same task compositions, only a split override differs → different group scores."""
        gc = override_results["groups"]["overlap_group_c"]["acc,none"]
        gd = override_results["groups"]["overlap_group_d"]["acc,none"]
        assert gc != pytest.approx(gd)

    def test_configs_reflect_override(self, override_results):
        """The overridden task's config should show the validation split."""
        cfg_c = override_results["configs"]["overlap_group_c::task_with_custom_df_3"]
        cfg_d = override_results["configs"]["overlap_group_d::task_with_custom_df_3"]
        assert cfg_c["test_split"] == "test"
        assert cfg_d["test_split"] == "validation"

    def test_both_namespaced_keys_present(self, override_results):
        r = override_results["results"]
        assert "overlap_group_c::task_with_custom_df_3" in r
        assert "overlap_group_d::task_with_custom_df_3" in r
        assert "overlap_group_c::task_with_custom_df_1" in r
        assert "overlap_group_d::task_with_custom_df_1" in r
