"""Exercise bootstrap resampling through real spawned worker processes."""

import multiprocessing as mp
import os
import random
from functools import partial
from pathlib import Path

import pytest

from lm_eval.api import metrics


def _mean_with_pid(xs, pid_dir):
    """Record the executing process without changing the metric result."""
    (Path(pid_dir) / str(os.getpid())).touch()
    return metrics.mean(xs)


@pytest.mark.parametrize("iters", [500, 1000, 1500, 2000, 2500])
def test_bootstrap_spawn_pool_matches_serial(iters, monkeypatch, tmp_path):
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
    metric = partial(_mean_with_pid, pid_dir=str(tmp_path))
    metric.__name__ = "mean_with_pid"
    parallel_stderr = metrics.bootstrap_stderr(metric, xs, iters=iters)

    worker_pids = {int(path.name) for path in tmp_path.iterdir()}
    assert worker_pids
    assert os.getpid() not in worker_pids

    monkeypatch.setenv("DISABLE_MULTIPROC", "1")
    serial_stderr = metrics.bootstrap_stderr(metric, xs, iters=iters)
    assert (tmp_path / str(os.getpid())).exists()

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
