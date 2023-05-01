"""Module with Cohere API-based language model."""

import os
import transformers
from tqdm import tqdm
import cohere
from lm_eval.base import BaseLM
from lm_eval import utils


class CohereLM(BaseLM):
    """Language model accessed via Cohere API."""

    REQ_CHUNK_SIZE = 20

    def __init__(
        self,
        model="medium",
        truncate="START",
        max_retries=100,
        timeout=30,
        disable_is_greedy_computation=False,
    ):
        """Language model accessed via Cohere API.

        The API is documented here:
        - https://docs.cohere.ai/reference/generate.
        - https://cohere-sdk.readthedocs.io/

        This class is based on the gpt3.py model with the OpenAI API.

        Imortant: in order to use this LM you need to set the environment variable
        COHERE_API_SECRET_KEY to your Cohere API key.

        :param model: str
            The type of Cohere model to be used, can be either `medium` or `xlarge`.
            Defaults to `medium`.
        :param truncate: str
            Directly passed to Cohere API. One of NONE|START|END to specify how the API will handle inputs longer than the maximum token length.

            Passing START will discard the start of the input. END will discard the end of the input. In both cases, input is discarded until the remaining input is exactly the maximum input token length for the model.

            If NONE is selected, when the input exceeds the maximum input token length an error will be returned.
        :param max_retries: int
            Maximum number of retries for each API call.
        :param timeout: int
            Timeout for each API call in seconds.
        :param disable_is_greedy_computation: bool
            If True, check of whether continuation is greedy is disabled. This is useful if
            you have empty context strings and you don't need to check if greedy.
            Otherwise the API would return an error for empty context strings
            because of the greedy check. Defaults to False. Instead of a boolean value,
            None will be returned for is_greedy.
        """
        super().__init__()

        self.model = model
        self.truncate = truncate
        self.disable_is_greedy_computation = disable_is_greedy_computation

        # Set up Cohere API client
        api_key = os.environ["COHERE_API_SECRET_KEY"]
        self.cohere_client = cohere.Client(
            api_key, max_retries=max_retries, timeout=timeout
        )

        # set prefix token for rolling window loglikelihood computation
        self.prefix_token = self.cohere_client.tokenize(text="\n").tokens[0]

    @property
    def eot_token_id(self):
        raise NotImplementedError()

    @property
    def vocab_size(self):
        raise NotImplementedError()

    @property
    def max_length(self):
        # max length in number of tokens
        # "Generation models support up to 2048 tokens."
        # from https://docs.cohere.ai/docs/tokens
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.cohere_client.tokenize(text=string).tokens

    def tok_decode(self, tokens):
        return self.cohere_client.detokenize(tokens).text

    def loglikelihood(self, requests):
        # Add dummy encodings of context and continuation (both set to None).
        # This adaptation is to remain similar to other LMs whilst reducing the
        # overall number of API calls. We can do this because the _loglikelihood_tokens()
        # methods does not use any tokenised/encoded context/continuation.
        new_reqs = list(zip(requests, [None] * len(requests), [None] * len(requests)))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):

        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token,  # only difference to standard base LM class
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows, disable_tqdm=True
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        """Compute log-likelihood of generating a continuation from a context.

        The log likelihood of continuation can be obtained directly from the API.

        :param requests: list
            A list with elements ((context, continuation), context_enc, continuation_enc)
            Where
            context: str
                Context string.
            continuation: str
                The continuation as tokens over which log likelihood
                will be calculated.
            *_enc: the encoded (tokenised) version of context und continuations.
        :return: list
            A list of pairs (logprob, isgreedy)
            logprob: float
                The log probability of `continuation`
            isgreedy:
                Whether `continuation` would be generated by greedy sampling from `context`
        """

        res = []

        # We create datastructure for the requests
        # that allows us to:
        # 1) avoid duplicate requests to minimise API calls
        # 2) reorder requests to start with the longest requests to
        #   have out-of-memory errors at the beginning of loop.
        #
        # We use the following two methods:
        #
        # re_ord.get_reordered(): returns
        #   a list of unique context, continuation pairs
        #   (where any split between context and contuation is considered
        #   identical). Ordered by descending length of context+continuation
        # re_ord.get_original(res): given an array res of the same
        #   len as get_reordered(), returns the original array with
        #   res array elements switched in for index matching
        #   original values.

        decoded_request_available = requests[0][0] is not None

        def _collate(val):
            # makes the reorderer sort by descending
            # length of context+continuation
            # note that tokens are by default not used here,
            # because unavailable. Thus using str length
            if decoded_request_available:
                contin_str = val[0][1]
                context_str = val[0][0]
                combined = context_str + contin_str
            else:
                # if only tokens are available tokens are used
                contin_enc = val[1]
                context_enc = val[2]
                combined = context_enc + contin_enc
            return -len(combined), tuple(combined)

        re_ord = utils.Reorderer(requests, _collate)

        # iterate over chunks (i.e. subsets) of reordered requests
        for chunk in tqdm(
            list(utils.chunks(re_ord.get_reordered(), self.REQ_CHUNK_SIZE)),
            disable=disable_tqdm,
        ):
            for decoded_request, context_enc, contin_enc in chunk:

                if decoded_request is not None:
                    (context, continuation) = decoded_request
                else:
                    # if only tokens are available
                    context = self.tok_decode(context_enc)
                    continuation = self.tok_decode(contin_enc)

                response = self.cohere_client.generate(
                    model=self.model,  # "medium" or "xlarge"
                    prompt=context + continuation,
                    max_tokens=0,
                    temperature=0.0,
                    return_likelihoods="ALL",
                    truncate=self.truncate,
                )

                # compute token lengths for downstream tasks
                continuation_tokens = self.cohere_client.tokenize(text=continuation)
                continuation_token_len = len(continuation_tokens.tokens)
                overall_token_len = len(response.generations[0].token_likelihoods)
                context_token_len = overall_token_len - continuation_token_len

                if not self.disable_is_greedy_computation:
                    # Check if greedy
                    #
                    # Cohere's API does not provide a logprobs argument
                    # (like OpenAI's), thus we need a second generation API call
                    # to check if the greedy continuation is the same as the
                    # evaluated continuation.
                    greedy_response = self.cohere_client.generate(
                        model=self.model,  # "medium" or "xlarge"
                        prompt=context,
                        max_tokens=continuation_token_len,
                        temperature=0.0,
                        return_likelihoods="NONE",
                        truncate=self.truncate,
                    )
                    is_greedy = continuation == greedy_response.generations[0].text
                else:
                    is_greedy = None

                # compute logprob of continuation
                regular_likelihoods = response.generations[0].token_likelihoods
                continuation_logprob = sum(
                    [
                        token.likelihood
                        for token in regular_likelihoods[context_token_len:]
                    ]
                )

                answer = continuation_logprob, is_greedy
                res.append(answer)

                # TODO: does this cache key logic make any sense?
                # The logic is copied from gpt3.py LM class.
                cache_key = (context, continuation)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        return re_ord.get_original(res)

    def greedy_until(self, requests):
        """Generate greedily until a stopping sequence

        :param requests: list
            A list of pairs (context, until)
            context: str
                Context string
            until: [str]
                The string sequences to generate until. These string sequences
                may each span across multiple tokens, or may be part of one token.
        :return: list
            A list of strings continuation
            continuation: str
                The generated continuation.
        """

        if not requests:
            return []
        res = []

        def _collate(x):
            return len(x[0]), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            """Iterable that returns sublists of xs of max len `size`
            and with identical until values.
            """
            ret = []
            lastuntil = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != lastuntil:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = x[1]
                ret.append(x)

            if ret:
                yield ret, lastuntil

        for chunk, until in tqdm(
            list(sameuntil_chunks(re_ord.get_reordered(), self.REQ_CHUNK_SIZE))
        ):
            for context, _ in chunk:
                response = self.cohere_client.generate(
                    model=self.model,  # "medium" or "xlarge"
                    prompt=context,
                    max_tokens=self.max_gen_toks,
                    temperature=0.0,
                    return_likelihoods="ALL",
                    truncate=self.truncate,
                    # end sequences are NOT included in returned text
                    end_sequences=until,
                )

                gen_text = response.generations[0].text

                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), gen_text)

                res.append(gen_text)

        return re_ord.get_original(res)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
