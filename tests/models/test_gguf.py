import unittest
from unittest.mock import patch

import requests

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
    def test_retry_backoff_is_exponential_and_capped(self):
        lm = GGUFLM(base_url, retry_delay=2, retry_max_delay=5)
        failures = [requests.ConnectionError("server restarting")] * 5
        with (
            patch.object(
                lm, "_post_request", side_effect=[*failures, FakeResponse({"ok": True})]
            ) as post,
            patch("lm_eval.models.gguf.time.sleep") as sleep,
        ):
            self.assertEqual(
                lm._post_with_retries(lm.completions_url, {}), {"ok": True}
            )
        self.assertEqual(post.call_count, 6)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list], [2, 4, 5, 5, 5]
        )

    def test_exhausted_retries_preserve_cause_without_a_final_sleep(self):
        lm = GGUFLM(base_url, max_retries=2)
        error = requests.ConnectionError("server crashed")
        with (
            patch.object(lm, "_post_request", side_effect=error) as post,
            patch("lm_eval.models.gguf.time.sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "after 3 attempts") as raised,
        ):
            lm._post_with_retries(lm.completions_url, {})
        self.assertIs(raised.exception.__cause__, error)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_client_errors_fail_immediately(self):
        lm = GGUFLM(base_url)
        for status in (400, 401, 403, 404, 422):
            response = requests.Response()
            response.status_code = status
            with (
                self.subTest(status=status),
                patch.object(lm, "_post_request", return_value=response) as post,
                patch("lm_eval.models.gguf.time.sleep") as sleep,
                self.assertRaises(requests.HTTPError),
            ):
                lm._post_with_retries(lm.completions_url, {})
            self.assertEqual(post.call_count, 1)
            sleep.assert_not_called()

    def test_transient_http_errors_honor_capped_retry_after(self):
        for status in (408, 429, 500, 502, 503, 504):
            lm = GGUFLM(base_url, retry_delay=1, retry_max_delay=10)
            response = requests.Response()
            response.status_code = status
            response.headers["Retry-After"] = "120"
            with (
                self.subTest(status=status),
                patch.object(
                    lm,
                    "_post_request",
                    side_effect=[response, FakeResponse({"ok": True})],
                ),
                patch("lm_eval.models.gguf.time.sleep") as sleep,
            ):
                self.assertEqual(
                    lm._post_with_retries(lm.completions_url, {}), {"ok": True}
                )
            sleep.assert_called_once_with(10)

    def test_zero_retries_makes_one_attempt(self):
        lm = GGUFLM(base_url, max_retries=0)
        with (
            patch.object(lm, "_post_request", side_effect=requests.Timeout()) as post,
            patch("lm_eval.models.gguf.time.sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "after 1 attempts"),
        ):
            lm._post_with_retries(lm.completions_url, {})
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_invalid_retry_configuration_is_rejected(self):
        for kwargs in (
            {"max_retries": -1},
            {"retry_delay": -1},
            {"retry_max_delay": "nan"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GGUFLM(base_url, **kwargs)

    def test_loglikelihood_scoring(self):
        fake_post, calls = make_fake_server(
            token_logprobs={97: -0.5, 98: -2.0}  # 'a', 'b'
        )
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post):
            lm = GGUFLM(base_url, parallel=1)
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
            lm = GGUFLM(base_url, parallel=1)
            res = lm.loglikelihood(llm_instances([("x ", "ab")]))
        # trailing space moved from context into the continuation
        self.assertEqual(res, [(-3.0, True)])
        biases = [c["logit_bias"][0][0] for c in calls["completions"]]
        self.assertEqual(biases, [ord(" "), ord("a"), ord("b")])

    def test_loglikelihood_empty_continuation(self):
        fake_post, calls = make_fake_server()
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post):
            lm = GGUFLM(base_url, parallel=1)
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
            lm = GGUFLM(base_url, parallel=1)
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

    def test_generation_sampling_kwargs_and_greedy_override(self):
        fake_post, calls = make_fake_server()
        gen_kwargs = {
            "until": ["STOP"],
            "max_new_tokens": 17,
            "do_sample": False,
            "temperature": 0.8,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.2,
            "repetition_penalty": 1.1,
            "top_p": 0.9,
            "top_k": 30,
            "min_p": 0.05,
            "seed": 1234,
        }
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post):
            lm = GGUFLM(base_url, parallel=1, temperature=0.7)
            lm.generate_until(llm_instances([("x", gen_kwargs)], "generate_until"))
        self.assertEqual(
            calls["completions"][0],
            {
                "prompt": "x",
                "stop": ["STOP"],
                "max_tokens": 17,
                "temperature": 0.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.2,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 30,
                "min_p": 0.05,
                "seed": 1234,
            },
        )
        self.assertEqual(gen_kwargs["temperature"], 0.8)
        self.assertIn("repetition_penalty", gen_kwargs)

    def test_generation_task_temperature_overrides_model_default(self):
        fake_post, calls = make_fake_server()
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post):
            lm = GGUFLM(base_url, parallel=1, temperature=0.7)
            lm.generate_until(
                llm_instances(
                    [("x", {}), ("y", {"temperature": 0.3, "do_sample": True})],
                    "generate_until",
                )
            )
        self.assertEqual([c["temperature"] for c in calls["completions"]], [0.7, 0.3])

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

    def test_api_key_sent_on_every_post_request(self):
        """api_key reaches /tokenize and /v1/completions as a Bearer header."""
        fake_post, _ = make_fake_server(token_logprobs={97: -1.0})
        with patch(
            "lm_eval.models.gguf.requests.post", side_effect=fake_post
        ) as mock_post:
            lm = GGUFLM(base_url, api_key="sekret", parallel=1)
            lm.loglikelihood(llm_instances([("x", "a")]))
        expected = {"Authorization": "Bearer sekret"}
        self.assertEqual(len(mock_post.call_args_list), 3)  # tokenize, tokenize, score
        for call in mock_post.call_args_list:
            self.assertEqual(call.kwargs["headers"], expected)

    def test_api_key_sent_on_props(self):
        """The /props slot probe carries the auth header too, or a keyed server makes
        parallelism detection silently fall back to serial requests.
        """
        with patch("lm_eval.models.gguf.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"total_slots": 2}
            lm = GGUFLM(base_url, api_key="sekret")
            self.assertEqual(lm._detect_total_slots(), 2)
        self.assertEqual(
            mock_get.call_args.kwargs["headers"],
            {"Authorization": "Bearer sekret"},
        )

    def test_no_api_key_sends_no_auth_header(self):
        fake_post, _ = make_fake_server()
        with patch(
            "lm_eval.models.gguf.requests.post", side_effect=fake_post
        ) as mock_post:
            lm = GGUFLM(base_url, parallel=1)
            lm._tokenize("x", add_special=True)
        self.assertIsNone(mock_post.call_args.kwargs["headers"])

    def test_header_replaces_bearer_default(self):
        """Same semantics as TemplateAPI: an explicit `header` dict replaces the
        Bearer default rather than merging with it.
        """
        fake_post, _ = make_fake_server()
        with patch(
            "lm_eval.models.gguf.requests.post", side_effect=fake_post
        ) as mock_post:
            lm = GGUFLM(
                base_url,
                api_key="sekret",
                header={"X-Api-Key": "custom"},
                parallel=1,
            )
            lm._tokenize("x", add_special=True)
        self.assertEqual(mock_post.call_args.kwargs["headers"], {"X-Api-Key": "custom"})


if __name__ == "__main__":
    unittest.main()
