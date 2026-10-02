"""Exercise bootstrap resampling through real spawned worker processes."""

import multiprocessing as mp
import random

import pytest

from lm_eval.api import metrics


@pytest.mark.parametrize("iters", [500, 1000, 1500, 2000, 2500])
def test_bootstrap_spawn_pool_matches_serial(iters, monkeypatch):
    # Capture the real spawn context before replacing mp.Pool. Two workers keep
    # this test bounded even on large CI machines; the workers are not mocked.
    spawn_pool = mp.get_context("spawn").Pool
    monkeypatch.setattr(mp, "Pool", lambda processes: spawn_pool(processes=2))
    monkeypatch.delenv("DISABLE_MULTIPROC", raising=False)

    observed_samples = []
    sample_stddev = metrics.sample_stddev

    def capture_samples(samples):
        observed_samples.append(list(samples))
        return sample_stddev(samples)

    monkeypatch.setattr(metrics, "sample_stddev", capture_samples)
    xs = [0, 1, 1, 0, 1, 0, 1, 1]
    parallel_stderr = metrics.bootstrap_stderr(metrics.mean, xs, iters=iters)

    monkeypatch.setenv("DISABLE_MULTIPROC", "1")
    serial_stderr = metrics.bootstrap_stderr(metrics.mean, xs, iters=iters)

    # Build an independent reference using the existing seed-per-chunk contract.
    expected_samples = []
    for index, start in enumerate(range(0, iters, 1000)):
        rng = random.Random(index)
        for _ in range(min(1000, iters - start)):
            sample = rng.choices(xs, k=len(xs))
            expected_samples.append(sum(sample) / len(sample))

    assert len(observed_samples[0]) == iters
    assert len(observed_samples[1]) == iters
    assert observed_samples[0] == expected_samples
    assert observed_samples[1] == expected_samples
    assert parallel_stderr == serial_stderr
