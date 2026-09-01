import unittest
from unittest.mock import patch

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
    def test_loglikelihood_scoring(self):
        fake_post, calls = make_fake_server(
            token_logprobs={97: -0.5, 98: -2.0}  # 'a', 'b'
        )
        with patch("lm_eval.models.gguf.requests.post", side_effect=fake_post):
            lm = GGUFLM(base_url, parallel=1)
            res = lm.loglikelihood(llm_instances([("x", "ab")]))
        self.assertEqual(res, [(-2.5, True)])

        # continuation tokenized separately, without special tokens;
        # context tokenized once, with special tokens (fake BOS prepended)
        tok_calls = calls["tokenize"]
        self.assertEqual(
            {(c["content"], c["add_special"]) for c in tok_calls},
            {("x", True), ("ab", False)},
        )
        # one scoring request per continuation token, with correct prompts
        # ([BOS, 'x'] ++ prefix) and logit bias on the target token
        prompts = [c["prompt"] for c in calls["completions"]]
        biases = [c["logit_bias"][0][0] for c in calls["completions"]]
        self.assertEqual(prompts, [[1, ord("x")], [1, ord("x"), ord("a")]])
        self.assertEqual(biases, [ord("a"), ord("b")])

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


if __name__ == "__main__":
    unittest.main()
