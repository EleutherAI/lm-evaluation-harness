from lm_eval.humaneval_eval import HumanEval
from lm_eval.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)


def custom_simple_evals(model_name="gpt-4o-mini", base_url="https://api.openai.com/v1"):
    debug = False
    
    models = {
        model_name: ChatCompletionSampler(
            model=model_name,
            base_url=base_url,
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        )
    }

    # grading_sampler = ChatCompletionSampler(model="gpt-4o")
    # equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name, debug_mode):
        # num_examples = (
        #     args.examples if args.examples is not None else (5 if debug_mode else None)
        # )
        # Set num_examples = None to reproduce full evals
        num_examples = None
        match eval_name:
            case "humaneval":
                return HumanEval(num_examples=10 if debug_mode else num_examples)
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {
        eval_name: get_evals(eval_name, debug)
        for eval_name in ["humaneval"]
    }
    # debug_suffix = "_DEBUG" if args.debug else ""
    # print(debug_suffix)
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            metrics = result.metrics | {"score": result.score}
    
    evaluation_results = {
        "model": model_name,
        "eval_name": eval_name,
        "metrics": metrics,
    }

    return evaluation_results