import copy
import os
from functools import lru_cache
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, cast

from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import eval_logger, simple_parse_args_string


class LogLikelihoodResult(NamedTuple):
    log_likelihood: float
    is_greedy: bool


def _verify_credentials(creds: Any) -> None:
    """
    Verifies that all required keys are present in the credentials dictionary.
    Args:
        creds (Any): A dictionary containing the credentials.
    Raises:
        ValueError: If any of the necessary credentials are missing, with guidance on which environment variables need to be set.
    """
    required_keys = ["apikey", "url", "project_id"]
    env_var_mapping = {
        "apikey": "WATSONX_API_KEY",
        "url": "WATSONX_URL",
        "project_id": "WATSONX_PROJECT_ID",
    }
    missing_keys = [key for key in required_keys if key not in creds or not creds[key]]

    if missing_keys:
        missing_env_vars = [env_var_mapping[key] for key in missing_keys]
        raise ValueError(
            f"Missing required credentials: {', '.join(missing_keys)}. Please set the following environment variables: {', '.join(missing_env_vars)}"
        )


@lru_cache(maxsize=None)
def get_watsonx_credentials() -> Dict[str, str]:
    """
    Retrieves Watsonx API credentials from environmental variables.
    Returns:
        Dict[str, str]: A dictionary containing the credentials necessary for authentication, including
                        keys such as `apikey`, `url`, and `project_id`.
    Raises:
        AssertionError: If the credentials format is invalid or any of the necessary credentials are missing.
    """

    credentials = {
        "apikey": os.getenv("WATSONX_API_KEY", None),
        "url": os.getenv("WATSONX_URL", None),
        "project_id": os.getenv("WATSONX_PROJECT_ID", None),
    }

    _verify_credentials(credentials)
    return credentials


@register_model("watsonx_llm")
class WatsonxLLM(LM):
    """
    Implementation of LM model interface for evaluating Watsonx model with the lm_eval framework.
    See https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md for reference.
    """

    @classmethod
    def create_from_arg_string(
        cls: Type["WatsonxLLM"],
        arg_string: str,
        additional_config: Optional[Dict] = None,
    ) -> "WatsonxLLM":
        """
        Allow the user to specify model parameters (TextGenerationParameters) in CLI arguments.
        """
        try:
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        except ImportError:
            raise ImportError(
                "Could not import ibm_watsonx_ai: Please install lm_eval[ibm_watsonx_ai] package."
            )

        args = simple_parse_args_string(arg_string)
        args.update(additional_config)

        model_id = args.pop("model_id", None)
        if model_id is None:
            raise ValueError("'model_id' is required, please pass it in 'model_args'")

        if not args.get("do_sample", None):
            args["temperature"] = None
            args["top_p"] = None
            args["top_k"] = None
            args["seed"] = None

        generate_params = {
            GenParams.DECODING_METHOD: (
                "greedy" if not args.get("do_sample", None) else "sample"
            ),
            GenParams.LENGTH_PENALTY: args.get("length_penalty", None),
            GenParams.TEMPERATURE: args.get("temperature", None),
            GenParams.TOP_P: args.get("top_p", None),
            GenParams.TOP_K: args.get("top_k", None),
            GenParams.RANDOM_SEED: args.get("seed", None),
            GenParams.REPETITION_PENALTY: args.get("repetition_penalty", None),
            GenParams.MIN_NEW_TOKENS: args.get("min_new_tokens", None),
            GenParams.MAX_NEW_TOKENS: args.get("max_new_tokens", 256),
            GenParams.STOP_SEQUENCES: args.get("stop_sequences", None),
            GenParams.TIME_LIMIT: args.get("time_limit", None),
            GenParams.TRUNCATE_INPUT_TOKENS: args.get("truncate_input_tokens", None),
            GenParams.RETURN_OPTIONS: {
                "generated_tokens": True,
                "input_tokens": True,
                "token_logprobs": True,
                "token_ranks": True,
            },
        }

        generate_params = {k: v for k, v in generate_params.items() if v is not None}

        return cls(
            watsonx_credentials=get_watsonx_credentials(),
            model_id=model_id,
            generate_params=generate_params,
        )

    def __init__(
        self,
        watsonx_credentials: Dict,
        model_id,
        generate_params: Optional[Dict[Any, Any]] = None,
    ) -> None:
        try:
            from ibm_watsonx_ai import APIClient
            from ibm_watsonx_ai.foundation_models import ModelInference
        except ImportError:
            raise ImportError(
                "Could not import ibm_watsonx_ai: Please install lm_eval[ibm_watsonx_ai] package."
            )
        super().__init__()
        client = APIClient(watsonx_credentials)
        project_id = watsonx_credentials.get("project_id", None)
        deployment_id = watsonx_credentials.get("deployment_id", None)
        client.set.default_project(project_id)
        self.generate_params = generate_params
        self.model = ModelInference(
            model_id=model_id,
            deployment_id=deployment_id,
            api_client=client,
            project_id=project_id,
        )
        self._model_id = model_id

    @staticmethod
    def _has_stop_token(response_tokens: List[str], context_tokens: List[str]) -> bool:
        """
        Determines whether a stop token has been generated in the `response_tokens` compared to the `context_tokens`.
        If the tokens do not match as expected, the function raises a RuntimeError, indicating a possible
        misalignment between the tokens generated by the tokenizer and the model.
        Args:
            response_tokens (List[str]): The List of tokens generated as a response by the model.
            context_tokens (List[str]): The List of tokens representing the input context.
        Returns:
            bool: True if the `response_tokens` likely contain a stop token that terminates the sequence,
                  otherwise raises an exception.
        Raises:
            RuntimeError: If there is an unexpected mismatch between the `response_tokens` and the `context_tokens`.
        """
        context_length = len(context_tokens)
        if response_tokens[: context_length - 1] == context_tokens[:-1]:
            return (
                response_tokens[-1] != context_tokens[-1]
            )  # only last token differs, probably stop sequence (</s>)
        raise RuntimeError(
            f"There is an unexpected difference between tokenizer and model tokens:\n"
            f"context_tokens={context_tokens}\n"
            f"response_tokens={response_tokens[:context_length]}"
        )

    def _check_model_logprobs_support(self):
        """
        Verifies if the model supports returning log probabilities for input tokens.
        This function sends a prompt to the model and checks whether the model's response
        includes log probabilities for the input tokens. If log probabilities are not present,
        it raises a `RuntimeError`, indicating that the model is not supported.
        Raises:
            RuntimeError: If the model does not return log probabilities for input tokens.
        """
        tokens = self.model.generate_text(
            prompt=["The best ice cream flavor is:"],
            params=self.generate_params,
            raw_response=True,
        )[0]["results"][0]
        if all(token.get("logprob", None) is None for token in tokens["input_tokens"]):
            raise RuntimeError(
                f"Model {self._model_id} is not supported: does not return logprobs for input tokens"
            )

    def _get_log_likelihood(
        self,
        input_tokens: List[Dict[str, float]],
        context_tokens: List[Dict[str, float]],
    ) -> LogLikelihoodResult:
        """
        Calculates the log likelihood of the generated tokens compared to the context tokens.
        Args:
            input_tokens (List[Dict[str, float]]): A List of token dictionaries, each containing
                token information like `text` and `logprob`.
            context_tokens (List[Dict[str, float]]): A List of token dictionaries representing
                the input context.
        Returns:
            LogLikelihoodResult: An object containing the calculated log likelihood and a boolean
            flag indicating if the tokens were generated greedily.
        """

        response_tokens = [token["text"] for token in input_tokens]
        context_length = len(context_tokens)

        if self._has_stop_token(response_tokens, context_tokens):
            context_length -= 1

        return LogLikelihoodResult(
            log_likelihood=sum(
                token.get("logprob", 0) for token in input_tokens[context_length:]
            ),
            is_greedy=all(
                token["rank"] == 1 for token in input_tokens[context_length:]
            ),
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generates text responses for a List of requests, with progress tracking and caching.
        Args:
            requests (List[Instance]): A List of instances, each containing a text input to be processed.
        Returns:
            List[str]: A List of generated responses.
        """
        requests = [request.args for request in requests]
        results = []

        for request in tqdm(
            requests,
            desc="Running generate_until function ...",
        ):
            context, continuation = request
            try:
                response = self.model.generate_text(context, self.generate_params)
            except Exception as exp:
                eval_logger.error("Error while generating text.")
                raise exp

            results.append(response)
            self.cache_hook.add_partial(
                "generate_until", (context, continuation), response
            )

        return results

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Args:
            requests: Each request contains Instance.args : Tuple[str, str] containing:
                1. an input string to the LM and
                2. a target string on which the loglikelihood of the LM producing this target,
                   conditioned on the input, will be returned.
        Returns:
            Tuple (loglikelihood, is_greedy) for each request according to the input order:
                loglikelihood: probability of generating the target string conditioned on the input
                is_greedy: True if and only if the target string would be generated by greedy sampling from the LM
        """
        try:
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        except ImportError:
            raise ImportError(
                "Could not import ibm_watsonx_ai: Please install lm_eval[ibm_watsonx_ai] package."
            )
        self._check_model_logprobs_support()
        generate_params = copy.copy(self.generate_params)
        generate_params[GenParams.MAX_NEW_TOKENS] = 1

        requests = [request.args for request in requests]
        results: List[LogLikelihoodResult] = []

        # Note: We're not using batching due to (current) indeterminism of loglikelihood values when sending batch of requests
        for request in tqdm(
            requests,
            desc="Running loglikelihood function ...",
        ):
            context, continuation = request
            try:
                tokenized_context = self.model.tokenize(
                    prompt=context, return_tokens=True
                )["result"]["tokens"]
            except Exception as exp:
                eval_logger.error("Error while model tokenize.")
                raise exp

            input_prompt = context + continuation

            try:
                response = self.model.generate_text(
                    prompt=input_prompt, params=generate_params, raw_response=True
                )
            except Exception as exp:
                eval_logger.error("Error while model generate text.")
                raise exp

            log_likelihood_response = self._get_log_likelihood(
                response["results"][0]["input_tokens"], tokenized_context
            )
            results.append(log_likelihood_response)
            self.cache_hook.add_partial(
                "loglikelihood",
                (context, continuation),
                (
                    log_likelihood_response.log_likelihood,
                    log_likelihood_response.is_greedy,
                ),
            )

        return cast(List[Tuple[float, bool]], results)

    def loglikelihood_rolling(self, requests) -> List[Tuple[float, bool]]:
        """
        Used to evaluate perplexity on a data distribution.
        Args:
            requests: Each request contains Instance.args : Tuple[str] containing an input string to the model whose
                entire loglikelihood, conditioned on purely the EOT token, will be calculated.
        Returns:
            Tuple (loglikelihood,) for each request according to the input order:
                loglikelihood: solely the probability of producing each piece of text given no starting input.
        """
        try:
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        except ImportError:
            raise ImportError(
                "Could not import ibm_watsonx_ai: Please install lm_eval[ibm_watsonx_ai] package."
            )
        self._check_model_logprobs_support()
        generate_params = copy.deepcopy(self.generate_params)
        generate_params[GenParams.MAX_NEW_TOKENS] = 1

        requests = [request.args for request in requests]
        results: List[LogLikelihoodResult] = []

        # Note: We're not using batching due to (current) indeterminism of loglikelihood values when sending batch of requests
        for request in tqdm(
            requests,
            desc="Running loglikelihood_rolling function ...",
        ):
            context, continuation = request
            try:
                response = self.model.generate_text(
                    prompt=context, params=generate_params, raw_response=True
                )
            except Exception as exp:
                eval_logger.error("Error while model generate text.")
                raise exp

            log_likelihood_response = self._get_log_likelihood(
                response["results"][0]["input_tokens"], []
            )
            results.append(log_likelihood_response)
            self.cache_hook.add_partial(
                "loglikelihood_rolling",
                (context, continuation),
                log_likelihood_response.log_likelihood,
            )

        return cast(List[Tuple[float, bool]], results)
