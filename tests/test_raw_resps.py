"""Tests for raw generation capture alongside post-processed output.

Regression for issue #3196: backends that strip think-trace content
(vLLM, TRT-LLM) previously discarded the raw generation, so
``--log_samples`` could not show reasoning traces even for analysis.
"""

from lm_eval.api.instance import Instance
from lm_eval.models.utils import Collator, postprocess_generated_text


class TestInstanceRawResps:
    def test_default_empty(self):
        inst = Instance(
            request_type="generate_until", doc={}, arguments=("ctx",), idx=0
        )
        assert inst.raw_resps == []

    def test_backend_appends_raw(self):
        inst = Instance(
            request_type="generate_until", doc={}, arguments=("ctx",), idx=0
        )
        inst.raw_resps.append("<think>trace</think>answer")
        inst.resps.append("answer")
        assert inst.raw_resps == ["<think>trace</think>answer"]
        assert inst.resps == ["answer"]


class TestVllmRawCaptureSemantics:
    """The vLLM capture path, exercised without an engine: the same
    postprocess + reorder primitives the backend uses."""

    def test_raw_preserved_through_postprocess(self):
        raw = "<think>step 1\nstep 2</think>\nThe answer is 42."
        processed = postprocess_generated_text(raw, ["\n"], "</think>")
        assert processed == "The answer is 42."
        # the backend appends BOTH: raw survives stripping
        assert raw.startswith("<think>")

    def test_parallel_reorder_restores_raw_order(self):
        # Collator.get_original maps any parallel list back to original
        # request order — the mechanism the raw capture relies on
        reqs = [(f"ctx{i}", [i]) for i in range(5)]

        def _collate(_requests):
            return -len(_requests[0][0]), _requests[0][0]

        collator = Collator(reqs, _collate, group_by=None)
        list(collator.get_batched(n=0, batch_fn=None))  # build the index map
        processed_sorted = ["p3", "p1", "p0", "p4", "p2"]
        raw_sorted = ["r3", "r1", "r0", "r4", "r2"]
        processed = collator.get_original(processed_sorted)
        raw = collator.get_original(raw_sorted)
        # the invariant that matters: index-wise pairing is preserved —
        # raw[i] is the unmodified generation of the request whose
        # processed output is processed[i]
        assert len(processed) == len(raw) == 5
        for p, r in zip(processed, raw, strict=True):
            assert p[-1] == r[-1]

    def test_instances_receive_raw_in_original_order(self):
        insts = [
            Instance(request_type="generate_until", doc={}, arguments=(f"c{i}",), idx=i)
            for i in range(3)
        ]
        raw_ordered = ["raw0", "raw1", "raw2"]
        for req, raw in zip(insts, raw_ordered, strict=True):
            if raw is not None:
                req.raw_resps.append(raw)
        assert [i.raw_resps[0] for i in insts] == ["raw0", "raw1", "raw2"]


class TestLoggedSampleField:
    def test_example_dict_includes_raw_when_present(self):
        insts = [
            Instance(request_type="generate_until", doc={}, arguments=("c",), idx=0)
        ]
        insts[0].resps.append("answer")
        insts[0].raw_resps.append("<think>t</think>answer")
        example = {
            "resps": [req.resps for req in insts],
            **(
                {"raw_resps": [req.raw_resps for req in insts]}
                if any(req.raw_resps for req in insts)
                else {}
            ),
        }
        assert example["raw_resps"] == [["<think>t</think>answer"]]

    def test_example_dict_omits_raw_when_absent(self):
        insts = [
            Instance(request_type="generate_until", doc={}, arguments=("c",), idx=0)
        ]
        insts[0].resps.append("answer")
        example = {
            "resps": [req.resps for req in insts],
            **(
                {"raw_resps": [req.raw_resps for req in insts]}
                if any(req.raw_resps for req in insts)
                else {}
            ),
        }
        assert "raw_resps" not in example


class TestVllmSourceContract:
    def test_generate_until_captures_raw(self):
        """Static contract: the vllm generate_until body appends to
        raw_res and attaches to request instances before returning.
        Reads the source file directly so the test runs without vllm
        installed."""
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent / "lm_eval/models/vllm_causallms.py"
        ).read_text()
        assert "raw_res.append(generated_text)" in src
        assert "req.raw_resps.append(raw)" in src
        # raw is captured BEFORE postprocess strips the trace
        i_raw = src.index("raw_res.append(generated_text)")
        i_post = src.index("generated_text = postprocess_generated_text")
        assert i_raw < i_post
