import json
import os
import time
import warnings
from typing import List, Optional, Union

import requests  # needs to be imported in order to create gigachat temp acess_token

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.utils import eval_logger


warnings.filterwarnings(
    "ignore"
)  # turn off insecure connection warning if verify_certificate=False


@register_model(
    "gigachat_llms",
)
class GigaChatAPI(LocalChatCompletion):
    def __init__(
        self,
        base_url=None,
        auth_url=None,  # authorization url to get acess_token
        verify_certificate=False,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            verify_certificate=verify_certificate,
            **kwargs,
        )
        self.expiration_time = 0
        self.auth_url = auth_url

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        if generate:
            temperature = gen_kwargs.pop("temperature", None)
            do_sample = gen_kwargs.pop("do_sample", None)

            if do_sample is not None:  # GigaChat API does not have do sample option.
                if not do_sample:  # Ensure greedy decoding if do_sample=False
                    gen_kwargs["repetition_penalty"] = 1.0
                    gen_kwargs["top_p"] = 0.0
                elif temperature == 0.0:
                    eval_logger.warning(
                        "You cannot set do_sample=True and temperature=0. Automatically setting temperature=1."
                    )
                    temperature = 1.0
            if (
                temperature == 0.0
            ):  # Ensure greedy decoding by setting top_p=0 and repetition_penalty = 1
                temperature = (
                    1.0  # temperature cannot be set to zero. Use top_p instead
                )
                gen_kwargs["repetition_penalty"] = 1.0
                gen_kwargs["top_p"] = 0.0
            print(
                {
                    "messages": messages,
                    "model": self.model,
                    "temperature": temperature,
                    **gen_kwargs,
                }
            )
            return {
                "messages": messages,
                "model": self.model,
                "temperature": temperature,
                **gen_kwargs,
            }
        else:
            return None

    @property  # Don't use cached_property as we need to check that the access_token has not expired.
    def header(self) -> dict:
        """Override this property to return the headers for the API request."""

        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    @property  # Don't use cached_property as we need to check that the acess_token has not expired.
    def api_key(self):
        self.key = os.environ.get(
            "GIGACHAT_CREDENTIALS", None
        )  # GigaChat access token.
        if self.key:
            return self.key  # If access token is available, return access token.
        RqUID = os.environ.get(
            "GIGACHAT_RQUID", None
        )  # Unique identification request. Complies with uuid4 format. Value must match regular expression (([0-9a-fA-F-])36)
        auth_token = os.environ.get(
            "GIGACHAT_TOKEN", None
        )  # Client Secret. Credential for GigaChat API.
        scope = os.environ.get(
            "SCOPE", None
        )  # type of your API. Possible values: [GIGACHAT_API_PERS, GIGACHAT_API_B2B, GIGACHAT_API_CORP].
        if not scope:
            scope = "GIGACHAT_API_PERS"
            eval_logger.warning(
                "SCOPE environment variable not found. Automatically set to GIGACHAT_API_PERS."
            )

        if RqUID is None or auth_token is None:
            raise ValueError(
                "Credentials not found. Please set GIGACHAT_RQUID and GIGACHAT_TOKEN environment variables."
            )
        if self.expiration_time == 0 or self.expiration_time < int(
            time.time() * 1000
        ):  # Check if the access token exists and is valid. If not, create a new one
            try:
                token_ = self._get_token_gigachat(RqUID, auth_token, scope)
                self.key, self.expiration_time = (
                    token_["access_token"],
                    token_["expires_at"],
                )
            except Exception as e:
                raise ValueError(
                    f"Invalid credentials: {e}. Please set correct GIGACHAT_RQUID and GIGACHAT_TOKEN environment variables. Or check that the SCOPE was set correctly."
                )
        return self.key

    def _get_token_gigachat(self, rqUID: str, auth_token: str, scope: str) -> str:
        """
        Creates temporal token using credentials.

        rqUID - Unique identification request. Complies with uuid4 format. Value must match regular expression (([0-9a-fA-F-])36)
        auth_token - Client Secret. Credential for GigaChat API.
        scope - type of your API. Possible values: [GIGACHAT_API_PERS, GIGACHAT_API_B2B, GIGACHAT_API_CORP].

        Returns an access token for authorizing API requests. The access token is valid for 30 minutes. Issue it if current time > expiration time.
        """

        payload = f"scope={scope}"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": rqUID,
            "Authorization": f"Basic {auth_token}",
        }

        response = requests.request(
            "POST",
            self.auth_url,
            headers=headers,
            data=payload,
            verify=False,
        )
        return json.loads(response.text)
