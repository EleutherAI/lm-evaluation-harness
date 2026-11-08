"""POLAR-Bench v2: only final answers are visible to either conversation peer."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks.polar_bench import utils


class ListDocs(list):
    @property
    def features(self):
        return dict.fromkeys(self[0]) if self else {}


def final_answer(response):
    """Visible text for non-reasoning models; reject unexpected incomplete tags.

    This defensive cleanup does not support reasoning-prefilled chat templates.
    """
    response = str(response or "")
    if "<think>" in response and response.rfind("<think>") > response.rfind("</think>"):
        return "", "unfinished_thinking"
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[1]
    response = utils.strip_internal_channel_tags(response).strip()
    return response, "final" if response else "empty_final"


def process_results(doc, results, model_a_seed_name="evaluated-model", base_seed=42):
    result = results[0]
    score = utils.score_example(
        doc,
        result["transcript"],
        model_a_seed_name,
        base_seed,
        True,
    )
    turns = result["turn_info"]
    n = len(turns)
    return {
        "privacy": score["privacy_score"],
        "utility": score["utility_score"],
        "overall": score["example_score"],
        "final_answer_rate": sum(t["status"] == "final" for t in turns) / n
        if n
        else 0.0,
        "zero_turn_rate": float(n == 0),
    }


class PolarBenchTask(ConfigurableTask):
    VERSION = 2
    MAX_MULTITURN_STEPS = 64

    def __init__(self, config=None, attacker=None, **kwargs):
        config = deepcopy(config or {})
        config.pop("class", None)
        # Config export deep-copies callables: a bound method would copy this
        # task and its live attacker HTTP client (which contains thread locks).
        config.setdefault("process_results", process_results)
        metadata = config.setdefault("metadata", {})
        settings = {
            "thinking_mode": os.getenv(
                "POLAR_THINKING_MODE", metadata.get("thinking_mode", "final_only")
            ),
            "model_a_seed_name": os.getenv(
                "POLAR_MODEL_A_NAME",
                metadata.get("model_a_seed_name", "evaluated-model"),
            ),
            "attacker_model": os.getenv(
                "POLAR_ATTACKER_MODEL",
                metadata.get("attacker_model", "meta-llama/Llama-3.3-70B-Instruct"),
            ),
            "base_seed": int(os.getenv("POLAR_SEED", metadata.get("base_seed", 42))),
            "max_rounds": int(
                os.getenv("POLAR_MAX_ROUNDS", metadata.get("max_rounds", 6))
            ),
            "attacker_max_tokens": int(
                os.getenv(
                    "POLAR_ATTACKER_MAX_TOKENS",
                    metadata.get("attacker_max_tokens", 32768),
                )
            ),
            "protocol": "final-answer-only-v2",
        }
        if settings["thinking_mode"] != "final_only":
            raise ValueError(
                "This integration supports only non-reasoning models (final_only)"
            )
        if not 1 <= settings["max_rounds"] <= self.MAX_MULTITURN_STEPS:
            raise ValueError("POLAR_MAX_ROUNDS must be between 1 and 64")
        if settings["attacker_max_tokens"] < 1:
            raise ValueError("POLAR_ATTACKER_MAX_TOKENS must be a positive integer")
        metadata.update(settings)
        self.settings = settings
        self.attacker = attacker
        super().__init__(config=config, **kwargs)

    def download(self, dataset_kwargs=None, **kwargs):
        path = os.getenv("POLAR_DATASET_PATH")
        metadata = self.config.metadata
        if path:
            metadata["dataset_source"] = "local"
        else:
            from huggingface_hub import hf_hub_download

            revision = metadata["dataset_revision"]
            path = hf_hub_download(
                repo_id="Qiaoyuan/POLAR-Bench",
                repo_type="dataset",
                filename="data/privacy_benchmark_rendered_repaired.json",
                revision=revision,
            )
            metadata["dataset_source"] = f"Qiaoyuan/POLAR-Bench@{revision}"
        raw = Path(path).read_bytes()
        metadata["dataset_sha256"] = hashlib.sha256(raw).hexdigest()
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise TypeError("POLAR data must be a JSON array")
        # Do not silently change the evaluated population by skipping broken rows.
        for row in rows:
            usable, reason = utils.is_usable_example(row)
            if not usable:
                raise ValueError(
                    f"Unusable sample {utils.extract_example_id(row)!r}: {reason}"
                )
        if not rows:
            raise ValueError("POLAR dataset is empty")
        metadata["dataset_rows"] = len(rows)
        self.dataset = {"test": ListDocs(rows)}

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["test"]

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        if num_fewshot:
            raise ValueError("POLAR requires num_fewshot=0")
        if kwargs.get("system_instruction"):
            raise ValueError("POLAR supplies its own per-sample system instruction")
        return ""

    def doc_to_text(self, doc):
        return utils.build_a_system(doc)

    def doc_to_target(self, doc):
        return ""

    def init_multiturn_state(
        self, doc, ctx, gen_kwargs, apply_chat_template=False, chat_template=None
    ):
        if not apply_chat_template or chat_template is None:
            raise ValueError(
                "POLAR requires --apply_chat_template to preserve the system role"
            )
        turns = utils.parse_attacker_turns(doc)
        budget = utils.extract_attack_turn_budget(doc, self.settings["max_rounds"])
        budget = min(len(turns), budget) if turns else budget
        if not 1 <= budget <= self.MAX_MULTITURN_STEPS:
            raise ValueError(
                "Invalid POLAR sample turn budget; refusing silent rollout truncation"
            )
        return {
            "doc": doc,
            "transcript": [],
            "turn_info": [],
            "turn": 0,
            "scripted": turns,
            "budget": budget,
            "adaptive": not turns or utils.extract_attack_strength(doc) == 5,
            "done": False,
            "stop_reason": None,
            "gen_kwargs": deepcopy(gen_kwargs),
            "chat_template": chat_template,
        }

    def multiturn_is_done(self, state):
        return state["done"]

    def _attacker(self):
        if self.attacker is None:
            from lm_eval.tasks.polar_bench.attacker import Attacker

            self.attacker = Attacker(self.settings["attacker_model"])
        return self.attacker

    def multiturn_next_request(self, state):
        if state["done"]:
            return None
        doc, turn = state["doc"], state["turn"]
        sample_id = utils.extract_example_id(doc)
        if state["adaptive"]:
            planned = state["scripted"][turn] if state["scripted"] else ""
            messages = utils.build_b_turn_messages(
                doc, state["transcript"], turn + 1, state["budget"], planned
            )
            attack = (
                self._attacker()
                .chat(
                    system_prompt=utils.build_b_system(doc),
                    messages=messages,
                    max_tokens=self.settings["attacker_max_tokens"],
                    seed=utils.stable_seed(
                        sample_id,
                        self.settings["model_a_seed_name"],
                        self.settings["attacker_model"],
                        "model_b",
                        "adaptive",
                        turn + 1,
                        base_seed=self.settings["base_seed"],
                    ),
                )
                .strip()
            )
            if attack == "<STOP>":
                state.update(done=True, stop_reason="attacker_stop")
                return None
        else:
            attack = state["scripted"][turn]
        state["transcript"].append({"role": "B", "content": attack})
        messages = utils.build_a_messages_from_transcript(
            utils.build_a_system(doc), state["transcript"]
        )
        try:
            prompt = state["chat_template"](messages, add_generation_prompt=True)
        except TypeError:
            prompt = state["chat_template"](messages)
        # Some backend renderers retry unsupported templates without the system
        # message. Never score a conversation after dropping the privacy policy.
        policy = messages[0]["content"]
        if not isinstance(prompt, str) or not (
            policy in prompt or json.dumps(policy, ensure_ascii=True)[1:-1] in prompt
        ):
            raise ValueError("Chat template did not preserve the POLAR system policy")
        generation = deepcopy(state["gen_kwargs"])
        generation["seed"] = utils.stable_seed(
            sample_id,
            self.settings["model_a_seed_name"],
            "model_a",
            "adaptive" if state["adaptive"] else "scripted",
            turn + 1 if state["adaptive"] else len(state["transcript"]),
            base_seed=self.settings["base_seed"],
        )
        return prompt, generation

    def multiturn_consume_response(self, state, response):
        answer, status = final_answer(response)
        state["transcript"].append({"role": "A", "content": answer})
        state["turn_info"].append({"status": status})
        state["turn"] += 1
        if state["turn"] >= state["budget"]:
            state.update(done=True, stop_reason="turn_budget")

    def multiturn_result(self, state):
        if not state["done"]:
            raise RuntimeError("POLAR rollout was interrupted by the harness step cap")
        return {
            "transcript": state["transcript"],
            "turn_info": state["turn_info"],
            "stop_reason": state["stop_reason"],
            "protocol": self.settings["protocol"],
        }

    def process_results(self, doc, results):
        return process_results(
            doc, results, self.settings["model_a_seed_name"], self.settings["base_seed"]
        )
