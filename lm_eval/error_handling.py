import time
from functools import wraps
from typing import Any, Callable, Optional


def retry_on_rate_limit_error(
    on_exception: Exception,
    max_retries: Optional[int] = None,
    backoff_time: float = 3.0,
    backoff_multiplier: float = 1.5,
    on_exception_callback: Optional[Callable[[Exception], Any]] = None,
):
    """Retry on an LLM Provider's rate limit error with exponential backoff
    For example, to use for OpenAI, do the following:
    ```
    from openai import RateLimitError

    # Recommend specifying max_retries to avoid infinite loops!
    @retry_on_rate_limit_error(RateLimitError, max_retries=3)
    def completion(...):
        # Wrap OpenAI completion function here
        ...
    ```
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sleep_time = backoff_time
            attempt = 0
            while max_retries is None or attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except on_exception as e:
                    if on_exception_callback is not None:
                        on_exception_callback(e)
                    time.sleep(sleep_time)
                    sleep_time *= backoff_multiplier
                    attempt += 1

        return wrapper

    return decorator
