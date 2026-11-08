"""Explicit, lazily connected attacker service. No calls during task discovery."""

import os

from lm_eval.tasks.polar_bench.task import final_answer


class Attacker:
    def __init__(self, model):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "POLAR protocol 5 requires the optional client: pip install openai"
            ) from exc

        url = os.getenv("POLAR_ATTACKER_BASE_URL")
        if not url:
            raise ValueError("Protocol 5 requires POLAR_ATTACKER_BASE_URL")
        timeout = float(os.getenv("POLAR_ATTACKER_TIMEOUT", "600"))
        if not 0 < timeout < float("inf"):
            raise ValueError("POLAR_ATTACKER_TIMEOUT must be positive and finite")
        self.model = model
        self.client = OpenAI(
            base_url=url,
            api_key=os.getenv("POLAR_ATTACKER_API_KEY", "EMPTY"),
            timeout=timeout,
            max_retries=2,
        )

    def chat(self, system_prompt, messages, max_tokens, seed):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError(
                f"Attacker reached its {max_tokens}-token limit; attack was not silently accepted"
            )
        text, status = final_answer(choice.message.content)
        if status != "final":
            raise RuntimeError("Attacker produced no complete visible attack")
        return text
