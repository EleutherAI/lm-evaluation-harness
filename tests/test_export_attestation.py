import json

from scripts.export_attestation import build_attestation


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_attestation_hashes_results_and_matching_samples(tmp_path):
    run_dir = tmp_path / "gpt2"
    run_dir.mkdir()
    results_path = run_dir / "results_2026-05-20T00-00-00.json"
    samples_path = run_dir / "samples_arc_easy_2026-05-20T00-00-00.jsonl"

    _write_json(
        results_path,
        {
            "model_name": "gpt2",
            "git_hash": "abc123",
            "task_hashes": {"arc_easy": "taskhash"},
            "results": {
                "arc_easy": {
                    "alias": "ARC Easy",
                    "acc,none": 0.25,
                    "acc_stderr,none": 0.01,
                    "acc_norm,none": 0.3,
                }
            },
        },
    )
    _write_jsonl(
        samples_path,
        [
            {
                "doc_id": 0,
                "doc_hash": "d0",
                "prompt_hash": "p0",
                "target_hash": "t0",
                "filtered_resps": ["A"],
            }
        ],
    )

    attestation = build_attestation(tmp_path)

    assert attestation["schema_version"] == "lm-eval-attestation-v1"
    assert attestation["model_id"] == "gpt2"
    assert attestation["tasks"] == ["arc_easy"]
    assert attestation["results"] == [
        {"task_id": "arc_easy", "metric": "acc,none", "value": 0.25, "stderr": 0.01},
        {"task_id": "arc_easy", "metric": "acc_norm,none", "value": 0.3},
    ]
    assert len(attestation["results_sha256"]) == 64
    assert len(attestation["outputs_sha256"]) == 64
    assert attestation["source"]["git_hash"] == "abc123"
    assert attestation["source"]["task_hashes"] == {"arc_easy": "taskhash"}


def test_results_hash_is_stable_across_json_key_order(tmp_path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()

    _write_json(
        run_a / "results_2026-05-20T00-00-00.json",
        {
            "model_name": "model",
            "results": {
                "task_b": {"metric,none": 2},
                "task_a": {"metric_stderr,none": 0.1, "metric,none": 1},
            },
        },
    )
    _write_json(
        run_b / "results_2026-05-20T00-00-00.json",
        {
            "results": {
                "task_a": {"metric,none": 1, "metric_stderr,none": 0.1},
                "task_b": {"metric,none": 2},
            },
            "model_name": "model",
        },
    )

    assert build_attestation(run_a)["results_sha256"] == build_attestation(run_b)["results_sha256"]


def test_outputs_hash_is_none_without_sample_logs(tmp_path):
    results_path = tmp_path / "results_2026-05-20T00-00-00.json"
    _write_json(results_path, {"model_name": "model", "results": {"task": {"acc,none": 1.0}}})

    attestation = build_attestation(results_path)

    assert attestation["outputs_sha256"] is None
    assert attestation["source"]["sample_paths"] == []
