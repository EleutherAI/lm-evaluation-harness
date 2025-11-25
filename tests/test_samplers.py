"""Tests for lm_eval.api.samplers module."""

from __future__ import annotations

import pytest

from lm_eval.api.samplers import (
    SAMPLER_REGISTRY,
    ContextSampler,
    FirstNSampler,
    get_sampler,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_docs() -> list[dict]:
    """A simple list of documents for testing."""
    return [
        {"id": 0, "text": "first"},
        {"id": 1, "text": "second"},
        {"id": 2, "text": "third"},
        {"id": 3, "text": "fourth"},
        {"id": 4, "text": "fifth"},
    ]


@pytest.fixture
def large_docs() -> list[dict]:
    """A larger list for testing sampling behavior."""
    return [{"id": i, "text": f"doc_{i}"} for i in range(100)]


# =============================================================================
# ContextSampler Tests
# =============================================================================


class TestContextSampler:
    """Tests for the default ContextSampler (random sampling)."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_sample_returns_exactly_n_documents(self, sample_docs, n):
        """Sampling n documents returns exactly n documents."""
        sampler = ContextSampler(sample_docs, rnd=42)

        result = sampler.sample(n=n)

        assert len(result) == n

    def test_sample_with_seed_is_reproducible(self, sample_docs):
        """Same seed produces same samples."""
        sampler1 = ContextSampler(sample_docs, rnd=42)
        sampler2 = ContextSampler(sample_docs, rnd=42)

        result1 = sampler1.sample(n=3)
        result2 = sampler2.sample(n=3)

        assert result1 == result2

    def test_different_seeds_produce_different_samples(self, large_docs):
        """Different seeds produce different samples (with high probability)."""
        sampler1 = ContextSampler(large_docs, rnd=42)
        sampler2 = ContextSampler(large_docs, rnd=999)

        result1 = sampler1.sample(n=10)
        result2 = sampler2.sample(n=10)

        assert result1 != result2

    def test_sample_zero_returns_empty(self, sample_docs):
        """Requesting 0 samples returns empty list."""
        sampler = ContextSampler(sample_docs, rnd=42)

        result = sampler.sample(n=0)

        assert result == []

    def test_sample_negative_raises(self, sample_docs):
        """Requesting negative samples raises AssertionError."""
        sampler = ContextSampler(sample_docs, rnd=42)

        with pytest.raises(AssertionError, match=">=0"):
            sampler.sample(n=-5)

    def test_sample_excludes_eval_doc(self, sample_docs):
        """When eval_doc is provided, it's excluded from results."""
        sampler = ContextSampler(sample_docs, rnd=42)
        eval_doc = sample_docs[0]

        result = sampler.sample(n=4, eval_doc=eval_doc)

        assert eval_doc not in result
        assert len(result) == 4

    def test_sample_all_docs_with_exclusion(self, sample_docs):
        """Can sample all other docs when excluding one."""
        sampler = ContextSampler(sample_docs, rnd=42)
        eval_doc = sample_docs[2]

        result = sampler.sample(n=4, eval_doc=eval_doc)

        assert len(result) == 4
        assert eval_doc not in result

    def test_fewshot_indices_filters_documents(self, sample_docs):
        """fewshot_indices parameter limits which docs are available."""
        indices = [0, 2, 4]  # Only use docs at positions 0, 2, 4
        sampler = ContextSampler(sample_docs, rnd=42, fewshot_indices=indices)

        # Force loading by calling fewshot_docs
        available = sampler.fewshot_docs()

        assert len(available) == 3
        assert available[0] == sample_docs[0]
        assert available[1] == sample_docs[2]
        assert available[2] == sample_docs[4]

    def test_set_rnd_changes_random_state(self, large_docs):
        """set_rnd allows changing the random seed after initialization."""
        sampler = ContextSampler(large_docs, rnd=42)
        first_sample = sampler.sample(n=5)

        sampler.set_rnd(42)  # Reset to same seed
        second_sample = sampler.sample(n=5)

        assert first_sample == second_sample

    def test_replace_df_updates_documents(self, sample_docs):
        """replace_df changes the document pool."""
        sampler = ContextSampler(sample_docs, rnd=42)
        new_docs = [{"id": 100}, {"id": 200}]

        sampler.replace_df(new_docs)
        result = sampler.sample(n=2)

        assert all(doc in new_docs for doc in result)

    def test_replace_df_resets_loaded_state(self, sample_docs):
        """replace_df resets _loaded flag so fewshot_indices can reapply."""
        indices = [0, 1]
        sampler = ContextSampler(sample_docs, rnd=42, fewshot_indices=indices)

        # First access loads and filters
        sampler.fewshot_docs()
        assert sampler._loaded is True

        # Replace resets
        sampler.replace_df(sample_docs)
        assert sampler._loaded is False

    def test_empty_df_raises_on_sample(self):
        """Sampling from empty document pool raises assertion error."""
        sampler = ContextSampler([], rnd=42)

        with pytest.raises(AssertionError, match="no documents available"):
            sampler.sample(n=1)

    def test_none_df_defaults_to_empty(self):
        """None df defaults to empty list."""
        sampler = ContextSampler(None, rnd=42)

        assert sampler.df == []

    def test_sample_with_df_override(self, sample_docs, large_docs):
        """Passing df to sample() uses those docs instead."""
        sampler = ContextSampler(sample_docs, rnd=42)

        result = sampler.sample(n=10, df=large_docs)

        assert len(result) == 10
        # After sampling, the df should be updated
        assert sampler.df == large_docs


class TestRmEvalDoc:
    """Tests for the static rm_eval_doc method."""

    def test_removes_matching_doc(self):
        """Removes the eval doc from the list."""
        docs = [{"id": 1}, {"id": 2}, {"id": 3}]
        eval_doc = {"id": 2}

        result = ContextSampler.rm_eval_doc(eval_doc, docs)

        assert eval_doc not in result
        assert len(result) == 2

    def test_limits_to_n_results(self):
        """With n parameter, returns at most n items."""
        docs = [{"id": i} for i in range(10)]
        eval_doc = {"id": 5}

        result = ContextSampler.rm_eval_doc(eval_doc, docs, n=3)

        assert len(result) == 3
        assert eval_doc not in result

    def test_no_match_returns_all(self):
        """If eval_doc not in list, returns all items."""
        docs = [{"id": 1}, {"id": 2}]
        eval_doc = {"id": 999}

        result = ContextSampler.rm_eval_doc(eval_doc, docs)

        assert len(result) == 2


# =============================================================================
# FirstNSampler Tests
# =============================================================================


class TestFirstNSampler:
    """Tests for FirstNSampler (deterministic first-n sampling)."""

    def test_returns_first_n_in_order(self, sample_docs):
        """Returns exactly the first n documents in original order."""
        sampler = FirstNSampler(sample_docs)

        result = sampler.sample(n=3)

        assert result == sample_docs[:3]
        assert result[0]["id"] == 0
        assert result[1]["id"] == 1
        assert result[2]["id"] == 2

    def test_is_deterministic(self, sample_docs):
        """Always returns same result regardless of seed."""
        sampler1 = FirstNSampler(sample_docs, rnd=42)
        sampler2 = FirstNSampler(sample_docs, rnd=999)

        result1 = sampler1.sample(n=3)
        result2 = sampler2.sample(n=3)

        assert result1 == result2

    def test_sample_all(self, sample_docs):
        """Can request all documents."""
        sampler = FirstNSampler(sample_docs)

        result = sampler.sample(n=5)

        assert result == sample_docs

    def test_exceeding_available_raises(self, sample_docs):
        """Requesting more than available raises assertion error."""
        sampler = FirstNSampler(sample_docs)

        with pytest.raises(AssertionError, match="exceeds"):
            sampler.sample(n=10)

    def test_ignores_eval_doc(self, sample_docs):
        """FirstNSampler ignores eval_doc parameter (returns first n regardless)."""
        sampler = FirstNSampler(sample_docs)
        eval_doc = sample_docs[0]

        result = sampler.sample(n=3, eval_doc=eval_doc)

        # FirstNSampler doesn't exclude eval_doc - it just returns first n
        assert result == sample_docs[:3]


# =============================================================================
# Registry Tests
# =============================================================================


class TestSamplerRegistry:
    """Tests for the sampler registry and get_sampler function."""

    def test_registry_contains_default(self):
        """Registry has 'default' sampler."""
        assert "default" in SAMPLER_REGISTRY
        assert SAMPLER_REGISTRY["default"] is ContextSampler

    def test_registry_contains_first_n(self):
        """Registry has 'first_n' sampler."""
        assert "first_n" in SAMPLER_REGISTRY
        assert SAMPLER_REGISTRY["first_n"] is FirstNSampler

    def test_get_sampler_returns_class(self):
        """get_sampler returns the sampler class, not instance."""
        result = get_sampler("default")

        assert result is ContextSampler
        assert isinstance(result, type)

    def test_get_sampler_unknown_raises_keyerror(self):
        """Unknown sampler name raises KeyError with helpful message."""
        with pytest.raises(KeyError, match="no sampling strategy"):
            get_sampler("nonexistent_sampler")

    def test_get_sampler_error_lists_available(self):
        """Error message includes available sampler names."""
        with pytest.raises(KeyError, match="default"):
            get_sampler("bad_name")


# =============================================================================
# Integration Tests
# =============================================================================


class TestSamplerIntegration:
    """Integration tests for typical usage patterns."""

    def test_method_chaining(self, sample_docs):
        """Methods can be chained together."""
        sampler = ContextSampler(sample_docs)

        result = sampler.set_rnd(42).replace_df(sample_docs).sample(n=2)

        assert len(result) == 2

    def test_sampler_from_registry(self, sample_docs):
        """Full workflow: get sampler from registry and use it."""
        sampler_cls = get_sampler("default")
        sampler = sampler_cls(sample_docs, rnd=42)

        result = sampler.sample(n=3)

        assert len(result) == 3

    def test_first_n_from_registry(self, sample_docs):
        """Full workflow: get FirstNSampler from registry."""
        sampler_cls = get_sampler("first_n")
        sampler = sampler_cls(sample_docs)

        result = sampler.sample(n=2)

        assert result == sample_docs[:2]
