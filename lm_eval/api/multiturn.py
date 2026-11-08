"""Batched, sequential episodes over the existing generate_until model interface."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING

from lm_eval.api.instance import Instance


if TYPE_CHECKING:
    from lm_eval.api.model import LM
    from lm_eval.api.task import Task

logger = logging.getLogger(__name__)


def run_multi_turn_rollout(
    lm: LM,
    eval_tasks: dict[str, Task],
    requests: list[Instance],
    apply_chat_template: bool = False,
    chat_template=None,
) -> None:
    """Run one independent episode per document/repeat, batching active turns.

    Only finished episodes are returned to the normal filtering/scoring pipeline.
    Neither task hooks nor model failures are retried here. A stalled task or a
    step limit raises rather than silently scoring a partial conversation.
    Distributed data parallelism is unsupported; model tensor parallelism works.
    """
    if lm.world_size != 1:
        raise NotImplementedError("multi_turn_generate requires world_size=1")

    # Cached placeholders carry no live clients, renderer, or conversation state.
    # Each repeat gets fresh state and a private copy of generation options.
    episodes = []
    for instance in requests:
        task = eval_tasks[instance.task_name]
        ctx, gen_kwargs = instance.arguments
        for _ in range(instance.repeats):
            state = task.init_multiturn_state(
                instance.doc,
                ctx,
                deepcopy(gen_kwargs),
                apply_chat_template=apply_chat_template,
                chat_template=chat_template,
            )
            cap = task.MAX_MULTITURN_STEPS
            if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
                raise ValueError("MAX_MULTITURN_STEPS must be a positive integer")
            episodes.append((task, instance, state, cap))

    # One request per active episode forms a wave. Consume the whole wave
    # before constructing any next turn, preserving sequential feedback.
    pending = [(episode, 0) for episode in episodes]
    while pending:
        active = []
        wave = []
        for episode, step in pending:
            task, instance, state, cap = episode
            if task.multiturn_is_done(state):
                continue
            arguments = task.multiturn_next_request(state)
            if arguments is None:
                if not task.multiturn_is_done(state):
                    raise RuntimeError(
                        f"{instance.task_name}: next request is None but episode is not done"
                    )
                continue
            if step >= cap:
                raise RuntimeError(
                    f"{instance.task_name}: multi-turn step cap ({cap}) reached before completion"
                )
            wave.append(
                Instance(
                    request_type="generate_until",
                    doc=instance.doc,
                    arguments=arguments,
                    idx=step,
                    metadata=(instance.task_name, instance.doc_id, 1),
                )
            )
            active.append((episode, step))
        if not wave:
            break
        logger.info("Running multi_turn_generate wave with %d requests", len(wave))
        responses = lm.generate_until(wave)
        for response, (episode, _step) in zip(responses, active, strict=True):
            task, _, state, _ = episode
            task.multiturn_consume_response(state, response)
        pending = [(episode, step + 1) for episode, step in active]

    # Publish results only after every episode finishes: a failed peer must not
    # leave apparently successful partial results in the normal scoring pipeline.
    completed = [
        (instance, task.multiturn_result(state))
        for task, instance, state, _ in episodes
    ]
    for instance, result in completed:
        instance.resps.append(result)
