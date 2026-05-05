"""Tests for scripts/export_attestation.py. Stdlib-only."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from scripts.export_attestation import (
    ATTESTATION_VERSION,
    AttestationError,
    ResultsError,
    SamplesError,
    _format_float,
    build_attestation,
    canonical_encode,
    compute_outputs_sha256,
    derive_meta,
    derive_model_id,
    derive_tasks,
    discover_pair,
    extract_results,
    main,
    pair_samples_for_results,
    sha256_hex,
)


DATE_ID = "2024-01-01T00-00-00.000000"
RESULTS_HASH = "f1070dd00dedbe06a9ccf354eacc3ea40ae6636de117378e96114aacaf948522"
OUTPUTS_HASH = "67d91f2214d6f512947e562697905cf4af96717121a302794c8256b0c6aa0e6d"


def _make_results_doc(model_source="hf", model_name="EleutherAI/pythia-70m"):
    return {
        "model_source": model_source,
        "model_name": model_name,
        "model_name_sanitized": model_name.replace("/", "__"),
        "results": {
            "arc_easy": {
                "name": "arc_easy",
                "alias": "arc_easy",
                "sample_len": 2,
                "acc,none": 0.5,
                "acc_stderr,none": 0.1,
                "acc_norm,none": 0.75,
                "acc_norm_stderr,none": "N/A",
            },
            "hellaswag": {
                "name": "hellaswag",
                "alias": "hellaswag",
                "sample_len": 2,
                "acc,none": 0.25,
                "acc_stderr,none": 0.05,
            },
        },
        "configs": {"arc_easy": {}, "hellaswag": {}},
        "versions": {"arc_easy": "1.0", "hellaswag": "1.0"},
        "n-shot": {"arc_easy": 0, "hellaswag": 0},
        "config": {"model": "hf", "model_args": "pretrained=EleutherAI/pythia-70m"},
        "git_hash": "abcd1234",
        "date": "2024-01-01T00:00:00",
    }


def _make_sample(doc_id, filtered_resps, filter_key="none"):
    return {
        "doc_id": doc_id,
        "doc": {"question": f"q{doc_id}"},
        "target": "A",
        "arguments": {"gen_args_0": {"arg_0": "ctx"}},
        "resps": filtered_resps,
        "filtered_resps": filtered_resps,
        "filter": filter_key,
        "metrics": ["acc"],
        "doc_hash": "h" * 64,
        "prompt_hash": "p" * 64,
        "target_hash": "t" * 64,
    }


def _write_fixture(tmp_path, with_samples=True):
    model_dir = tmp_path / "EleutherAI__pythia-70m"
    model_dir.mkdir()
    results_path = model_dir / f"results_{DATE_ID}.json"
    results_path.write_text(json.dumps(_make_results_doc()), encoding="utf-8")

    if with_samples:
        for task_id, samples in [
            ("arc_easy", [_make_sample(0, ["A"]), _make_sample(1, ["B"])]),
            ("hellaswag", [_make_sample(0, ["X"]), _make_sample(1, ["Y"])]),
        ]:
            p = model_dir / f"samples_{task_id}_{DATE_ID}.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
    return model_dir


# A reference encoder that does not use json.dumps. Used to verify that the
# canonical encoding produces the same bytes when implemented from scratch.

def _hand_format_float(x):
    if not math.isfinite(x):
        raise ValueError("non-finite")
    if x == 0.0:
        return "0.0"
    s = repr(float(x))
    if "e" in s:
        mantissa, _, exp = s.partition("e")
        if exp.startswith("+"):
            exp = exp[1:]
        sign = "-" if exp.startswith("-") else ""
        digits = exp.lstrip("+-").lstrip("0") or "0"
        s = f"{mantissa}e{sign}{digits}"
    return s


def _hand_canonicalize(obj):
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return _hand_format_float(obj)
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    if isinstance(obj, list):
        return [_hand_canonicalize(x) for x in obj]
    if isinstance(obj, dict):
        return {
            unicodedata.normalize("NFC", k): _hand_canonicalize(v)
            for k, v in obj.items()
        }
    raise TypeError(type(obj).__name__)


def _hand_encode_str(s):
    out = ['"']
    for ch in s:
        cp = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif cp < 0x20 or cp >= 0x7F:
            if cp <= 0xFFFF:
                out.append(f"\\u{cp:04x}")
            else:
                cp -= 0x10000
                high = 0xD800 + (cp >> 10)
                low = 0xDC00 + (cp & 0x3FF)
                out.append(f"\\u{high:04x}\\u{low:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _hand_encode(obj):
    if obj is None:
        return "null"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, str):
        return _hand_encode_str(obj)
    if isinstance(obj, list):
        return "[" + ",".join(_hand_encode(x) for x in obj) + "]"
    if isinstance(obj, dict):
        items = sorted(obj.items())
        return "{" + ",".join(
            f"{_hand_encode_str(k)}:{_hand_encode(v)}" for k, v in items
        ) + "}"
    raise TypeError(type(obj).__name__)


def hand_canonical_encode(obj):
    return _hand_encode(_hand_canonicalize(obj)).encode("utf-8")


@pytest.mark.parametrize("value, expected", [
    (0.0, "0.0"),
    (-0.0, "0.0"),
    (1.0, "1.0"),
    (0.5, "0.5"),
    (-3.14, "-3.14"),
    (1e10, "10000000000.0"),
    (1e16, "1e16"),
    (1e-10, "1e-10"),
    (5e-324, "5e-324"),
    (-2.5e100, "-2.5e100"),
])
def test_format_float(value, expected):
    assert _format_float(value) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_format_float_rejects_non_finite(value):
    with pytest.raises(ValueError):
        _format_float(value)


class TestCanonicalEncoding:
    def test_sort_keys_and_separators(self):
        assert canonical_encode({"b": 2, "a": 1}) == b'{"a":1,"b":2}'

    def test_nested_sort_keys(self):
        assert canonical_encode({"outer": {"z": 1, "a": 2}}) == b'{"outer":{"a":2,"z":1}}'

    def test_floats_stringified(self):
        assert canonical_encode({"x": 0.5}) == b'{"x":"0.5"}'

    def test_disallows_nan(self):
        with pytest.raises(ValueError):
            canonical_encode({"x": float("nan")})

    def test_disallows_infinity(self):
        with pytest.raises(ValueError):
            canonical_encode({"x": float("inf")})

    def test_unicode_escaped(self):
        encoded = canonical_encode({"x": "café"})
        assert encoded == b'{"x":"caf\\u00e9"}'

    def test_nfc_normalizes_input(self):
        nfc = "café"
        nfd = "café"
        assert nfc != nfd
        assert canonical_encode({"x": nfc}) == canonical_encode({"x": nfd})

    def test_nfc_normalizes_dict_keys(self):
        nfc = "café"
        nfd = "café"
        assert canonical_encode({nfc: 1}) == canonical_encode({nfd: 1})

    def test_compact_no_whitespace(self):
        assert b" " not in canonical_encode([1, 2, {"a": 3}])

    def test_rejects_unsupported_types(self):
        with pytest.raises(TypeError):
            canonical_encode({"x": object()})

    def test_dict_keys_must_be_strings(self):
        with pytest.raises(TypeError, match="dict keys"):
            canonical_encode({1: "value"})


class TestResultsExtraction:
    def test_extracts_metric_filter_value_stderr(self):
        entries = extract_results(_make_results_doc())
        keys = {(e["task_id"], e["metric"], e["filter"]) for e in entries}
        assert keys == {
            ("arc_easy", "acc", "none"),
            ("arc_easy", "acc_norm", "none"),
            ("hellaswag", "acc", "none"),
        }

    def test_skips_alias_name_sample_len(self):
        for e in extract_results(_make_results_doc()):
            assert e["metric"] not in {"name", "alias", "sample_len"}

    def test_handles_na_stderr(self):
        entries = extract_results(_make_results_doc())
        acc_norm = next(
            e for e in entries if e["task_id"] == "arc_easy" and e["metric"] == "acc_norm"
        )
        assert acc_norm["stderr"] is None

    def test_value_is_canonical_string(self):
        entries = extract_results(_make_results_doc())
        acc = next(
            e for e in entries if e["task_id"] == "arc_easy" and e["metric"] == "acc"
        )
        assert acc["value"] == "0.5"
        assert acc["stderr"] == "0.1"

    def test_skips_non_scalar_value(self):
        doc = _make_results_doc()
        doc["results"]["arc_easy"]["text_metric,none"] = "some string"
        metrics = {(e["task_id"], e["metric"]) for e in extract_results(doc)}
        assert ("arc_easy", "text_metric") not in metrics

    def test_skips_bool_value(self):
        doc = _make_results_doc()
        doc["results"]["arc_easy"]["bool_metric,none"] = True
        metrics = {(e["task_id"], e["metric"]) for e in extract_results(doc)}
        assert ("arc_easy", "bool_metric") not in metrics

    def test_sort_order_stable(self):
        doc = _make_results_doc()
        rev = {
            "model_source": doc["model_source"],
            "model_name": doc["model_name"],
            "results": {
                k: dict(reversed(list(v.items())))
                for k, v in reversed(list(doc["results"].items()))
            },
        }
        assert extract_results(doc) == extract_results(rev)


class TestDeriveModelId:
    def test_uses_top_level_fields(self):
        doc = {"model_source": "hf", "model_name": "EleutherAI/pythia-70m"}
        assert derive_model_id(doc) == "hf/EleutherAI/pythia-70m"

    def test_falls_back_to_config_string(self):
        doc = {"config": {"model": "vllm", "model_args": "pretrained=foo/bar,dtype=auto"}}
        assert derive_model_id(doc) == "vllm/foo/bar"

    def test_falls_back_to_config_dict(self):
        doc = {"config": {"model": "vllm", "model_args": {"pretrained": "foo/bar"}}}
        assert derive_model_id(doc) == "vllm/foo/bar"

    def test_unknown_when_nothing_present(self):
        assert derive_model_id({}) == "unknown"

    def test_peft_takes_precedence(self):
        doc = {"config": {"model": "hf", "model_args": "pretrained=base/model,peft=adapter/lora"}}
        assert derive_model_id(doc) == "hf/adapter/lora"

    def test_delta_takes_precedence(self):
        doc = {"config": {"model": "hf", "model_args": {"pretrained": "base/model", "delta": "tweaked/model"}}}
        assert derive_model_id(doc) == "hf/tweaked/model"

    def test_strips_whitespace(self):
        doc = {"config": {"model": "hf", "model_args": "pretrained= foo/bar ,dtype=auto"}}
        assert derive_model_id(doc) == "hf/foo/bar"

    def test_empty_value_falls_through(self):
        doc = {"config": {"model": "hf", "model_args": "pretrained=,model=actual/name"}}
        assert derive_model_id(doc) == "hf/actual/name"

    def test_empty_dict_value_falls_through(self):
        doc = {"config": {"model": "hf", "model_args": {"pretrained": "", "model": "actual/name"}}}
        assert derive_model_id(doc) == "hf/actual/name"


def test_derive_meta():
    meta = derive_meta(_make_results_doc())
    assert meta == {
        "git_hash": "abcd1234",
        "versions": {"arc_easy": "1.0", "hellaswag": "1.0"},
        "n_shot": {"arc_easy": 0, "hellaswag": 0},
        "date": "2024-01-01T00:00:00",
    }


def test_derive_tasks_sorted_unique():
    entries = [
        {"task_id": "z", "metric": "m", "filter": "none", "value": "0.0", "stderr": None},
        {"task_id": "a", "metric": "m", "filter": "none", "value": "0.0", "stderr": None},
        {"task_id": "a", "metric": "n", "filter": "none", "value": "0.0", "stderr": None},
    ]
    assert derive_tasks(entries) == ["a", "z"]


class TestOutputsHash:
    def test_filtered_resps_only(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        samples_paths = sorted(model_dir.glob("samples_*.jsonl"))
        before = compute_outputs_sha256(samples_paths)

        # mutate fields that should NOT affect the digest
        for p in samples_paths:
            mutated = []
            for line in p.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                rec["resps"] = ["DIFFERENT"]
                rec["doc"] = {"question": "totally different"}
                rec["target"] = "Z"
                rec["arguments"] = {"gen_args_0": {"arg_0": "x"}}
                mutated.append(json.dumps(rec))
            p.write_text("\n".join(mutated) + "\n", encoding="utf-8")

        assert before == compute_outputs_sha256(samples_paths)

    def test_filtered_resps_change_diverges(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        samples_paths = sorted(model_dir.glob("samples_*.jsonl"))
        before = compute_outputs_sha256(samples_paths)

        target = next(p for p in samples_paths if "arc_easy" in p.name)
        lines = target.read_text(encoding="utf-8").splitlines()
        rec = json.loads(lines[0])
        rec["filtered_resps"] = ["CHANGED"]
        lines[0] = json.dumps(rec)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

        assert before != compute_outputs_sha256(samples_paths)

    def test_doc_id_ordering_invariant(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        samples_paths = sorted(model_dir.glob("samples_*.jsonl"))
        before = compute_outputs_sha256(samples_paths)

        for p in samples_paths:
            lines = p.read_text(encoding="utf-8").splitlines()
            p.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")

        assert before == compute_outputs_sha256(samples_paths)

    def test_collision_raises(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        target = model_dir / f"samples_arc_easy_{DATE_ID}.jsonl"
        rec = _make_sample(0, ["A"])
        target.write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n", encoding="utf-8")
        with pytest.raises(SamplesError, match="collision"):
            compute_outputs_sha256([target])

    def test_missing_required_keys(self, tmp_path):
        target = tmp_path / f"samples_x_{DATE_ID}.jsonl"
        target.write_text(json.dumps({"doc_id": 0}) + "\n", encoding="utf-8")
        with pytest.raises(SamplesError, match="missing"):
            compute_outputs_sha256([target])

    def test_missing_samples_file(self, tmp_path):
        with pytest.raises(SamplesError, match="not found"):
            compute_outputs_sha256([tmp_path / f"samples_x_{DATE_ID}.jsonl"])

    def test_malformed_json(self, tmp_path):
        target = tmp_path / f"samples_x_{DATE_ID}.jsonl"
        target.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(SamplesError):
            compute_outputs_sha256([target])


class TestDiscoverPair:
    def test_pairs_by_date_id(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        results_path, samples_paths = discover_pair(model_dir)
        assert results_path.name == f"results_{DATE_ID}.json"
        assert len(samples_paths) == 2
        assert all(DATE_ID in p.name for p in samples_paths)

    def test_picks_latest_results(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        older = model_dir / "results_2023-01-01T00-00-00.000000.json"
        older.write_text(json.dumps(_make_results_doc()), encoding="utf-8")
        results_path, _ = discover_pair(model_dir)
        assert DATE_ID in results_path.name

    def test_missing_dir(self, tmp_path):
        with pytest.raises(ResultsError, match="not a directory"):
            discover_pair(tmp_path / "does_not_exist")

    def test_empty_dir(self, tmp_path):
        with pytest.raises(ResultsError, match="no results_"):
            discover_pair(tmp_path)

    def test_pair_samples_skips_unrelated(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        results_path = model_dir / f"results_{DATE_ID}.json"
        other = model_dir / "samples_arc_easy_1999-01-01T00-00-00.000000.jsonl"
        other.write_text(json.dumps(_make_sample(0, ["A"])) + "\n", encoding="utf-8")
        paths = pair_samples_for_results(results_path)
        assert other not in paths


class TestBuildAttestation:
    def test_full_bundle(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        results_path, samples_paths = discover_pair(model_dir)
        bundle = build_attestation(results_path, samples_paths)
        assert bundle["harness_attestation_version"] == ATTESTATION_VERSION
        assert bundle["model_id"] == "hf/EleutherAI/pythia-70m"
        assert bundle["tasks"] == ["arc_easy", "hellaswag"]
        assert len(bundle["results"]) == 3
        assert len(bundle["results_sha256"]) == 64
        assert bundle["outputs_sha256"] is not None
        assert bundle["meta"]["git_hash"] == "abcd1234"

    def test_results_entries_are_strings(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        results_path, samples_paths = discover_pair(model_dir)
        bundle = build_attestation(results_path, samples_paths)
        for entry in bundle["results"]:
            assert isinstance(entry["value"], str)
            assert entry["stderr"] is None or isinstance(entry["stderr"], str)

    def test_no_samples_yields_null(self, tmp_path):
        model_dir = _write_fixture(tmp_path, with_samples=False)
        results_path = model_dir / f"results_{DATE_ID}.json"
        bundle = build_attestation(results_path, samples_paths=[])
        assert bundle["outputs_sha256"] is None
        assert bundle["results_sha256"] is not None

    def test_include_outputs_false(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        results_path, samples_paths = discover_pair(model_dir)
        bundle = build_attestation(results_path, samples_paths, include_outputs=False)
        assert bundle["outputs_sha256"] is None

    def test_meta_changes_dont_affect_results_sha(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        results_path, samples_paths = discover_pair(model_dir)
        bundle_a = build_attestation(results_path, samples_paths)

        doc = json.loads(results_path.read_text(encoding="utf-8"))
        doc["git_hash"] = "deadbeef"
        doc["date"] = "2099-12-31T00:00:00"
        results_path.write_text(json.dumps(doc), encoding="utf-8")

        bundle_b = build_attestation(results_path, samples_paths)
        assert bundle_a["results_sha256"] == bundle_b["results_sha256"]
        assert bundle_a["meta"]["git_hash"] != bundle_b["meta"]["git_hash"]

    def test_empty_results_raises(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        results_path = model_dir / f"results_{DATE_ID}.json"
        results_path.write_text(
            json.dumps({"model_source": "hf", "model_name": "x", "results": {}}),
            encoding="utf-8",
        )
        with pytest.raises(ResultsError, match="no scalar"):
            build_attestation(results_path, [])

    def test_missing_results_file_raises(self, tmp_path):
        with pytest.raises(ResultsError):
            build_attestation(tmp_path / "missing.json", None)


class TestCLI:
    def test_input_dir_to_stdout(self, tmp_path, capsys):
        model_dir = _write_fixture(tmp_path)
        rc = main(["--input_dir", str(model_dir)])
        assert rc == 0
        bundle = json.loads(capsys.readouterr().out)
        assert bundle["model_id"] == "hf/EleutherAI/pythia-70m"
        assert bundle["outputs_sha256"] is not None

    def test_results_to_output_file(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        results_path = model_dir / f"results_{DATE_ID}.json"
        out_path = tmp_path / "bundle.json"
        rc = main(["--results", str(results_path), "--output", str(out_path)])
        assert rc == 0
        bundle = json.loads(out_path.read_text(encoding="utf-8"))
        assert bundle["model_id"] == "hf/EleutherAI/pythia-70m"

    def test_no_samples_flag(self, tmp_path, capsys):
        model_dir = _write_fixture(tmp_path)
        rc = main(["--input_dir", str(model_dir), "--no_samples"])
        assert rc == 0
        bundle = json.loads(capsys.readouterr().out)
        assert bundle["outputs_sha256"] is None

    def test_missing_input_dir(self, tmp_path, capsys):
        rc = main(["--input_dir", str(tmp_path / "nope")])
        assert rc == 3
        assert "error" in capsys.readouterr().err

    def test_mutex_input_dir_and_results(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            main(["--input_dir", str(tmp_path), "--results", str(tmp_path / "x.json")])
        assert exc.value.code == 2

    def test_one_of_required(self):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2

    def test_samples_without_results(self, tmp_path, capsys):
        rc = main(["--input_dir", str(tmp_path), "--samples", str(tmp_path / "s.jsonl")])
        assert rc == 2
        assert "--samples requires --results" in capsys.readouterr().err

    def test_subprocess_invocation(self, tmp_path):
        # End-to-end: catches packaging issues that in-process calls miss.
        model_dir = _write_fixture(tmp_path)
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "scripts" / "export_attestation.py"
        result = subprocess.run(
            [sys.executable, str(script), "--input_dir", str(model_dir)],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        bundle = json.loads(result.stdout)
        assert bundle["results_sha256"] == RESULTS_HASH
        assert bundle["outputs_sha256"] == OUTPUTS_HASH


def test_schema_file_exists_and_parses():
    schema_path = Path(__file__).resolve().parents[2] / "scripts" / "attestation.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["type"] == "object"
    assert "harness_attestation_version" in schema["required"]


class TestCanonicalSpec:
    # The "recompute via stdlib json" tests catch parameter drift in
    # canonical_encode. The hand-canonical tests are the real cross-impl check
    # because they don't go through json.dumps at all.

    def test_results_sha_recompute_via_stdlib(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        results_path, samples_paths = discover_pair(model_dir)
        bundle = build_attestation(results_path, samples_paths)
        claim = {
            "model_id": bundle["model_id"],
            "tasks": bundle["tasks"],
            "results": bundle["results"],
        }
        canonical = json.dumps(
            claim, sort_keys=True, ensure_ascii=True,
            allow_nan=False, separators=(",", ":"),
        ).encode("utf-8")
        assert hashlib.sha256(canonical).hexdigest() == bundle["results_sha256"]

    def test_outputs_sha_recompute_via_stdlib(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        results_path, samples_paths = discover_pair(model_dir)
        bundle = build_attestation(results_path, samples_paths)

        by_task = {}
        for p in samples_paths:
            stem = p.name[: -len(".jsonl")]
            task_id = stem.split("_", 1)[1].rsplit("_", 1)[0]
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    by_task.setdefault(task_id, []).append(json.loads(line))

        h = hashlib.sha256()
        for task_id in sorted(by_task):
            for r in sorted(by_task[task_id], key=lambda r: (r["doc_id"], r.get("filter", ""))):
                payload = json.dumps(
                    {
                        "task_id": task_id,
                        "doc_id": r["doc_id"],
                        "filter": r.get("filter", ""),
                        "filtered_resps": r["filtered_resps"],
                    },
                    sort_keys=True, ensure_ascii=True,
                    allow_nan=False, separators=(",", ":"),
                ).encode("utf-8")
                h.update(payload)
                h.update(b"\n")
        assert h.hexdigest() == bundle["outputs_sha256"]

    def test_hand_canonical_matches_for_tiny_input(self):
        claim = {
            "model_id": "hf/x",
            "tasks": ["t"],
            "results": [
                {"task_id": "t", "metric": "acc", "filter": "none",
                 "value": "0.5", "stderr": "0.1"},
            ],
        }
        assert hand_canonical_encode(claim) == canonical_encode(claim)

    def test_hand_canonical_matches_for_realistic_fixture(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        results_path, samples_paths = discover_pair(model_dir)
        bundle = build_attestation(results_path, samples_paths)
        claim = {
            "model_id": bundle["model_id"],
            "tasks": bundle["tasks"],
            "results": bundle["results"],
        }
        hand = hand_canonical_encode(claim)
        assert hand == canonical_encode(claim)
        assert hashlib.sha256(hand).hexdigest() == bundle["results_sha256"]

    def test_hand_canonical_handles_nfc_and_floats(self):
        obj = {
            "name": "café",  # NFD; should normalize to NFC
            "values": [1.0, 1e16, -0.0, 0.1],
        }
        assert hand_canonical_encode(obj) == canonical_encode(obj)

    def test_canonical_byte_form_is_stable(self):
        claim = {
            "model_id": "hf/x",
            "tasks": ["t"],
            "results": [
                {"task_id": "t", "metric": "acc", "filter": "none",
                 "value": "0.5", "stderr": "0.1"},
            ],
        }
        assert canonical_encode(claim) == (
            b'{"model_id":"hf/x",'
            b'"results":[{"filter":"none","metric":"acc","stderr":"0.1",'
            b'"task_id":"t","value":"0.5"}],'
            b'"tasks":["t"]}'
        )

    def test_results_hash_golden(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        results_path, samples_paths = discover_pair(model_dir)
        bundle = build_attestation(results_path, samples_paths)
        assert bundle["results_sha256"] == RESULTS_HASH

    def test_outputs_hash_golden(self, tmp_path):
        model_dir = _write_fixture(tmp_path)
        results_path, samples_paths = discover_pair(model_dir)
        bundle = build_attestation(results_path, samples_paths)
        assert bundle["outputs_sha256"] == OUTPUTS_HASH


def test_exit_codes():
    assert ResultsError.exit_code == 3
    assert SamplesError.exit_code == 4
    assert issubclass(ResultsError, AttestationError)
    assert issubclass(SamplesError, AttestationError)


def test_sha256_hex_helper():
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
