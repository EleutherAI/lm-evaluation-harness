import unittest
from unittest.mock import patch

from requests.exceptions import RequestException

from lm_eval.api.instance import Instance
from lm_eval.models.gguf import GGUFLM


base_url = "http://fake-server:8080"


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def make_fake_server(token_logprobs=None, top_token_id=None):
    """Build a fake llama.cpp server.

    Tokenization is byte-level: each UTF-8 byte is a token id; a fake BOS
    (id 1) is prepended when add_special is true. Scoring requests (prompt
    given as a token-id array) return `token_logprobs[target_id]` as the
    sampled logprob (default -1.0), and a top_logprobs list whose argmax is
    `top_token_id` if given, else the target itself.
    """
    calls = {"tokenize": [], "completions": []}

    def fake_post(url, json=None, timeout=None, **kwargs):
        if url.endswith("/tokenize"):
            calls["tokenize"].append(json)
            ids = list(json["content"].encode("utf-8"))
            if json.get("add_special"):
                ids = [1] + ids
            return FakeResponse({"tokens": ids})
        elif url.endswith("/completions"):
            calls["completions"].append(json)
            prompt = json["prompt"]
            if isinstance(prompt, list):
                # scoring request
                target_id = json["logit_bias"][0][0]
                lp = (token_logprobs or {}).get(target_id, -1.0)
                top_id = top_token_id if top_token_id is not None else target_id
                top_lp = lp + 1.0 if top_id != target_id else lp
                entry = {
                    "id": target_id,
                    "token": "x",
                    "logprob": lp,
                    "top_logprobs": [
                        {"id": top_id, "token": "y", "logprob": top_lp},
                        {"id": target_id, "token": "x", "logprob": lp},
                    ],
                }
                return FakeResponse(
                    {"choices": [{"text": "x", "logprobs": {"content": [entry]}}]}
                )
            # generation request
            stop = json.get("stop")
            return FakeResponse({"choices": [{"text": f"generated text until {stop}"}]})
        raise AssertionError(f"unexpected url {url}")

    return fake_post, calls


def make_probe_server(probe_logprobs):
    """Build a fake llama.cpp server whose logprob-check requests drift.

    Scoring requests sent with `cache_prompt: False` (the start-of-run
    logprob check) return the next value of `probe_logprobs` as the sampled
    logprob; all other requests behave as in `make_fake_server`.
    """
    fake_post, calls = make_fake_server()
    values = iter(probe_logprobs)

    def probe_post(url, json=None, timeout=None, **kwargs):
        resp = fake_post(url, json, timeout)
        if url.endswith("/completions") and json.get("cache_prompt") is False:
            data = resp.json()
            data["choices"][0]["logprobs"]["content"][0]["logprob"] = next(values)
            return FakeResponse(data)
        return resp

    return probe_post, calls


def probe_calls(calls):
    """Return the logprob-check requests (sent with `cache_prompt: False`)."""
    return [c for c in calls["completions"] if c.get("cache_prompt") is False]


# logprobs returned by a drifting server for the 5 check requests (spread 2.0)
WOBBLY = [-1.0, -1.5, -2.0, -2.5, -3.0]


def llm_instances(args_list, request_type="loglikelihood"):
    return [
        Instance(
            request_type=request_type,
            doc=args,
            arguments=args if request_type == "loglikelihood" else (args[0], args[1]),
            idx=i,
        )
        for i, args in enumerate(args_list)
    ]


class GGUFLMTest(unittest.TestCase):
    def test_loglikelihood_scoring(self):
        fake_post, calls = make_fake_server(
            token_logprobs={97: -0.5, 98: -2.0}  # 'a', 'b'
        )
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post):
            lm = GGUFLM(base_url, parallel=1, logprob_check="off")
            res = lm.loglikelihood(llm_instances([("x", "ab")]))
        self.assertEqual(res, [(-2.5, True)])

        # context+continuation encoded together and split at the context
        # boundary (matches the HF backend); context also encoded separately;
        # both with special tokens (fake BOS prepended)
        tok_calls = calls["tokenize"]
        self.assertEqual(
            {(c["content"], c["add_special"]) for c in tok_calls},
            {("x", True), ("xab", True)},
        )
        # one scoring request per continuation token, with correct prompts
        # ([BOS, 'x'] ++ prefix) and logit bias on the target token
        prompts = [c["prompt"] for c in calls["completions"]]
        biases = [c["logit_bias"][0][0] for c in calls["completions"]]
        self.assertEqual(prompts, [[1, ord("x")], [1, ord("x"), ord("a")]])
        self.assertEqual(biases, [ord("a"), ord("b")])

    def test_loglikelihood_migrates_trailing_whitespace(self):
        fake_post, calls = make_fake_server()
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post):
            lm = GGUFLM(base_url, parallel=1, logprob_check="off")
            res = lm.loglikelihood(llm_instances([("x ", "ab")]))
        # trailing space moved from context into the continuation
        self.assertEqual(res, [(-3.0, True)])
        biases = [c["logit_bias"][0][0] for c in calls["completions"]]
        self.assertEqual(biases, [ord(" "), ord("a"), ord("b")])

    def test_loglikelihood_empty_continuation(self):
        fake_post, calls = make_fake_server()
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post):
            lm = GGUFLM(base_url, parallel=1, logprob_check="off")
            res = lm.loglikelihood(llm_instances([("x", ""), ("x", "a")]))
        # empty continuation scores (0.0, True) with no scoring requests
        self.assertEqual(res[0], (0.0, True))
        self.assertEqual(res[1], (-1.0, True))
        self.assertEqual(len(calls["completions"]), 1)

    def test_loglikelihood_non_greedy(self):
        fake_post, _ = make_fake_server(top_token_id=999)
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post):
            lm = GGUFLM(base_url, parallel=1)
            res = lm.loglikelihood(llm_instances([("x", "ab")]))
        self.assertEqual(res, [(-2.0, False)])

    def test_loglikelihood_forced_token_mismatch_raises(self):
        fake_post, _ = make_fake_server()

        def bad_post(url, json=None, timeout=None, **kwargs):
            resp = fake_post(url, json, timeout)
            if url.endswith("/completions") and isinstance(json["prompt"], list):
                data = resp.json()
                data["choices"][0]["logprobs"]["content"][0]["id"] = 424242
                return FakeResponse(data)
            return resp

        with patch("lm_eval.models.gguf.requests.post", side_effect=bad_post):
            lm = GGUFLM(base_url, parallel=1, logprob_check="off")
            with self.assertRaises(RuntimeError):
                lm.loglikelihood(llm_instances([("x", "a")]))

    def test_tokenize_cached(self):
        fake_post, calls = make_fake_server()
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post):
            lm = GGUFLM(base_url, parallel=1)
            lm.loglikelihood(llm_instances([("x", "a"), ("x", "b")]))
        ctx_calls = [c for c in calls["tokenize"] if c["content"] == "x"]
        self.assertEqual(len(ctx_calls), 1)

    def test_generate_until(self):
        fake_post, _ = make_fake_server()
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post):
            lm = GGUFLM(base_url, parallel=1)
            requests = [
                Instance(
                    request_type="generate_until",
                    doc={"input": doc},
                    arguments=(doc, {"until": stop}),
                    idx=i,
                )
                for i, (doc, stop) in enumerate(
                    [("input1", "stop1"), ("input2", "stop2")]
                )
            ]
            res = lm.generate_until(requests)
        self.assertEqual(
            res, ["generated text until stop1", "generated text until stop2"]
        )

    def test_generate_until_max_token_aliases(self):
        fake_post, calls = make_fake_server()
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post):
            lm = GGUFLM(base_url, parallel=1)
            requests = [
                Instance(
                    request_type="generate_until",
                    doc={"input": doc},
                    arguments=(doc, gen_kwargs),
                    idx=i,
                )
                for i, (doc, gen_kwargs) in enumerate(
                    [
                        ("a", {"max_tokens": "512"}),  # CLI-style string alias
                        ("b", {"max_gen_toks": 8, "max_tokens": "512"}),
                        ("c", {}),
                    ]
                )
            ]
            lm.generate_until(requests)
        max_tokens = [c.get("max_tokens") for c in calls["completions"]]
        # alias resolved and int()-converted; max_gen_toks wins over aliases;
        # backend default (256) when the task specifies nothing
        self.assertEqual(max_tokens, [512, 8, 256])

    def test_parallel_mapping_preserves_order(self):
        lm = GGUFLM(base_url, parallel=3)
        items = list(range(20))
        res = lm._map_requests(lambda x: x * 2, items, disable_tqdm=True)
        self.assertEqual(res, [x * 2 for x in items])

    def test_assign_slots_by_context(self):
        # consecutive same-context requests share a slot; groups round-robin
        args = [
            ("ctx1", "a"),
            ("ctx1", "b"),
            ("ctx2", "c"),
            ("ctx3", "d"),
            ("ctx2", "e"),
        ]
        self.assertEqual(GGUFLM._assign_slots_by_context(args, 2), [0, 0, 1, 0, 1])
        # parallel=1 pins everything to slot 0
        self.assertEqual(GGUFLM._assign_slots_by_context(args, 1), [0, 0, 0, 0, 0])

    def test_detect_total_slots(self):
        lm = GGUFLM(base_url)
        with patch("lm_eval.models.gguf.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"total_slots": 4}
            self.assertEqual(lm._detect_total_slots(), 4)
            self.assertEqual(lm._resolve_parallel(), 4)

        from requests.exceptions import RequestException

        lm = GGUFLM(base_url)
        with patch(
            "lm_eval.models.gguf.requests.get",
            side_effect=RequestException("no server"),
        ):
            self.assertIsNone(lm._detect_total_slots())
            # falls back to serial requests
            self.assertEqual(lm._resolve_parallel(), 1)

    def test_completions_url_derivation(self):
        for base, expected in [
            ("http://localhost:8080", "http://localhost:8080/v1/completions"),
            ("http://localhost:8080/", "http://localhost:8080/v1/completions"),
            ("http://localhost:8080/v1", "http://localhost:8080/v1/completions"),
            (
                "http://localhost:8080/v1/completions",
                "http://localhost:8080/v1/completions",
            ),
        ]:
            lm = GGUFLM(base, parallel=1)
            self.assertEqual(lm.completions_url, expected)
            self.assertEqual(lm.server_url, "http://localhost:8080")

    def test_logprob_check_steady_server(self):
        fake_post, calls = make_fake_server()
        with (
            patch("lm_eval.models.gguf.requests.post", side_effect=fake_post),
            patch("lm_eval.models.gguf.requests.get") as mock_get,
            self.assertNoLogs("lm_eval.models.gguf", level="WARNING"),
        ):
            lm = GGUFLM(base_url, parallel=1)
            res = lm.loglikelihood(llm_instances([("x", "ab")]))
        self.assertEqual(res, [(-2.0, True)])
        # 5 identical check requests for the first answer token, with prompt
        # caching off and no slot pinning, then the 2 real scoring requests
        probe = {
            "prompt": [1, ord("x")],
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": 2,
            "logit_bias": [[ord("a"), 100]],
            "cache_prompt": False,
        }
        self.assertEqual(calls["completions"][:5], [probe] * 5)
        self.assertEqual(len(calls["completions"]), 7)
        # /props is only read to build the failure message
        mock_get.assert_not_called()

    def test_logprob_check_wobbly_server_warns(self):
        fake_post, _ = make_probe_server(WOBBLY)
        with (
            patch("lm_eval.models.gguf.requests.post", side_effect=fake_post),
            patch("lm_eval.models.gguf.requests.get") as mock_get,
            self.assertLogs("lm_eval.models.gguf", level="WARNING") as logs,
        ):
            mock_get.return_value.json.return_value = {"build_info": "b1-abc"}
            lm = GGUFLM(base_url, parallel=1)
            res = lm.loglikelihood(llm_instances([("x", "a")]))
        self.assertEqual(len(logs.records), 1)
        message = logs.records[0].getMessage()
        self.assertIn("identical scoring requests", message)
        self.assertIn("spread 2.0000", message)
        self.assertIn("b1-abc", message)
        # the run keeps going and real scoring is unaffected
        self.assertEqual(res, [(-1.0, True)])

    def test_logprob_check_error_mode_raises(self):
        fake_post, calls = make_probe_server(WOBBLY)
        with (
            patch("lm_eval.models.gguf.requests.post", side_effect=fake_post),
            patch(
                "lm_eval.models.gguf.requests.get",
                side_effect=RequestException("no server"),
            ),
        ):
            lm = GGUFLM(base_url, parallel=1, logprob_check="error")
            with self.assertRaises(RuntimeError) as ctx:
                lm.loglikelihood(llm_instances([("x", "a")]))
            self.assertIn("identical scoring requests", str(ctx.exception))
            self.assertIn("unknown", str(ctx.exception))
            # stopped after the check, before any real scoring request
            self.assertEqual(len(calls["completions"]), 5)
            # the check ran once already, so a second call scores normally
            res = lm.loglikelihood(llm_instances([("x", "a")]))
        self.assertEqual(res, [(-1.0, True)])
        self.assertEqual(len(probe_calls(calls)), 5)

    def test_logprob_check_tolerance(self):
        cases = [
            # (probe logprobs, tolerance kwargs, expect warning)
            ([-1.0] * 4 + [-1.5], {"logprob_check_tol": 0.5}, False),
            ([-1.0] * 4 + [-1.5], {"logprob_check_tol": 0.25}, True),
            # spread 2**-7, just under the default tolerance of 0.01
            ([-1.0] * 4 + [-1.0078125], {}, False),
        ]
        for probe_logprobs, tol_kwargs, expect_warning in cases:
            fake_post, _ = make_probe_server(probe_logprobs)
            logs = (
                self.assertLogs("lm_eval.models.gguf", level="WARNING")
                if expect_warning
                else self.assertNoLogs("lm_eval.models.gguf", level="WARNING")
            )
            with (
                patch("lm_eval.models.gguf.requests.post", side_effect=fake_post),
                patch("lm_eval.models.gguf.requests.get"),
                logs,
            ):
                lm = GGUFLM(base_url, parallel=1, **tol_kwargs)
                lm.loglikelihood(llm_instances([("x", "a")]))

    def test_logprob_check_runs_once(self):
        fake_post, calls = make_fake_server()
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post):
            lm = GGUFLM(base_url, parallel=1)
            lm.loglikelihood(llm_instances([("x", "a")]))
            lm.loglikelihood(llm_instances([("y", "b")]))
        self.assertEqual(len(probe_calls(calls)), 5)

    def test_real_scoring_requests_unchanged(self):
        args = [("x", "ab"), ("y", "c")]
        fake_post_on, calls_on = make_fake_server()
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post_on):
            GGUFLM(base_url, parallel=1).loglikelihood(llm_instances(args))
        fake_post_off, calls_off = make_fake_server()
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post_off):
            GGUFLM(base_url, parallel=1, logprob_check="off").loglikelihood(
                llm_instances(args)
            )
        real_on = calls_on["completions"][5:]
        self.assertEqual(real_on, calls_off["completions"])
        for call in real_on + calls_off["completions"]:
            self.assertNotIn("cache_prompt", call)

    def test_logprob_check_off_sends_no_probe(self):
        fake_post, calls = make_probe_server(WOBBLY)
        with (
            patch("lm_eval.models.gguf.requests.post", side_effect=fake_post),
            self.assertNoLogs("lm_eval.models.gguf", level="WARNING"),
        ):
            lm = GGUFLM(base_url, parallel=1, logprob_check="off")
            lm.loglikelihood(llm_instances([("x", "a")]))
        for call in calls["completions"]:
            self.assertNotIn("cache_prompt", call)

    def test_generate_until_skips_logprob_check(self):
        fake_post, calls = make_probe_server(WOBBLY)
        with (
            patch("lm_eval.models.gguf.requests.post", side_effect=fake_post),
            self.assertNoLogs("lm_eval.models.gguf", level="WARNING"),
        ):
            lm = GGUFLM(base_url, parallel=1)
            lm.generate_until(
                [
                    Instance(
                        request_type="generate_until",
                        doc={"input": "input1"},
                        arguments=("input1", {"until": "stop1"}),
                        idx=0,
                    )
                ]
            )
        # only the generation request; no token-id scoring requests
        self.assertEqual(len(calls["completions"]), 1)
        self.assertIsInstance(calls["completions"][0]["prompt"], str)

    def test_logprob_check_skips_empty_answers(self):
        fake_post, calls = make_fake_server()
        with (
            patch("lm_eval.models.gguf.requests.post", side_effect=fake_post),
            self.assertNoLogs("lm_eval.models.gguf", level="WARNING"),
        ):
            lm = GGUFLM(base_url, parallel=1)
            res = lm.loglikelihood(llm_instances([("x", "")]))
        self.assertEqual(res, [(0.0, True)])
        self.assertEqual(len(calls["completions"]), 0)

    def test_logprob_check_settings(self):
        # no network access in __init__, so no patches needed
        with self.assertRaises(ValueError):
            GGUFLM(base_url, logprob_check="maybe")
        for tol in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                GGUFLM(base_url, logprob_check_tol=tol)
        # --model_args turns "true"/"false" into booleans
        self.assertEqual(GGUFLM(base_url, logprob_check=True).logprob_check, "warn")
        self.assertEqual(GGUFLM(base_url, logprob_check=False).logprob_check, "off")

    def test_get_model_info(self):
        # (a) check never ran and /props is unreachable
        lm = GGUFLM(base_url)
        with patch(
            "lm_eval.models.gguf.requests.get",
            side_effect=RequestException("no server"),
        ):
            self.assertEqual(
                lm.get_model_info(),
                {"llama_server_build": "", "logprob_check": None},
            )
        # a failed /props read is not cached; a later call retries it
        with patch("lm_eval.models.gguf.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"build_info": "b1-abc"}
            self.assertEqual(lm.get_model_info()["llama_server_build"], "b1-abc")

        # (b) after a run on a drifting server
        fake_post, _ = make_probe_server(WOBBLY)
        with (
            patch("lm_eval.models.gguf.requests.post", side_effect=fake_post),
            patch("lm_eval.models.gguf.requests.get") as mock_get,
        ):
            mock_get.return_value.json.return_value = {"build_info": "b1-abc"}
            lm = GGUFLM(base_url, parallel=1)
            with self.assertLogs("lm_eval.models.gguf", level="WARNING"):
                lm.loglikelihood(llm_instances([("x", "a")]))
            info = lm.get_model_info()
        self.assertEqual(
            info,
            {
                "llama_server_build": "b1-abc",
                "logprob_check": {
                    "mode": "warn",
                    "repeats": 5,
                    "spread": 2.0,
                    "tolerance": 0.01,
                    "within_tolerance": False,
                },
            },
        )

    def test_logprob_check_uses_first_non_empty_answer(self):
        fake_post, calls = make_fake_server()
        with (
            patch("lm_eval.models.gguf.requests.post", side_effect=fake_post),
            patch("lm_eval.models.gguf.requests.get") as mock_get,
        ):
            lm = GGUFLM(base_url, parallel=1)
            res = lm.loglikelihood(llm_instances([("x", ""), ("y", "b")]))
        self.assertEqual(res, [(0.0, True), (-1.0, True)])
        # the empty first answer is skipped; the probe uses the second request
        probes = probe_calls(calls)
        self.assertEqual(len(probes), 5)
        for probe in probes:
            self.assertEqual(probe["prompt"], [1, ord("y")])
            self.assertEqual(probe["logit_bias"], [[ord("b"), 100]])
        mock_get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
