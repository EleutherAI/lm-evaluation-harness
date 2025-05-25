import hashlib
import logging
import os
from functools import wraps
from typing import Callable, List, Optional, Union


eval_logger = logging.getLogger(__name__)


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

OVERRIDE_PATH = os.getenv("LM_HARNESS_CACHE_PATH")


PATH = OVERRIDE_PATH if OVERRIDE_PATH else f"{MODULE_DIR}/.cache"

# This should be sufficient for uniqueness
HASH_INPUT = "EleutherAI-lm-evaluation-harness"

HASH_PREFIX = hashlib.sha256(HASH_INPUT.encode("utf-8")).hexdigest()

FILE_SUFFIX = f".{HASH_PREFIX}.pickle"


def load_from_cache(file_name: str, cache: bool = False):
    if not cache:
        return
    try:
        import dill

        path = f"{PATH}/{file_name}{FILE_SUFFIX}"

        with open(path, "rb") as file:
            cached_task_dict = dill.loads(file.read())
            return cached_task_dict

    except Exception:
        eval_logger.debug(f"{file_name} is not cached, generating...")
        pass


def save_to_cache(file_name, obj):
    import dill

    if not os.path.exists(PATH):
        os.mkdir(PATH)

    file_path = f"{PATH}/{file_name}{FILE_SUFFIX}"

    eval_logger.debug(f"Saving {file_path} to cache...")
    with open(file_path, "wb") as file:
        file.write(dill.dumps(obj))


# NOTE the "key" param is to allow for flexibility
def delete_cache(key: str = ""):
    files = os.listdir(PATH)

    for file in files:
        if file.startswith(key) and file.endswith(FILE_SUFFIX):
            file_path = f"{PATH}/{file}"
            os.unlink(file_path)


def _build_cache_key(
    task: str,
    num_fewshot: int,
    rank: int,
    world_size: int,
    apply_chat_template: bool,
    fewshot_as_multiturn: bool,
    system_instruction: Optional[str],
    tokenizer_name: str,
) -> str:
    """Build cache key from parameters"""
    cache_key = f"requests-{task}-{num_fewshot}shot-rank{rank}-world_size{world_size}"

    if apply_chat_template:
        cache_key += "-chat_template"
    if fewshot_as_multiturn:
        cache_key += "-fewshot_as_multiturn"
    if system_instruction is not None:
        # Import utils here to avoid circular imports
        import utils

        cache_key += f"-system_prompt_hash{utils.hash_string(system_instruction)}"
    cache_key += f"-tokenizer{tokenizer_name}"

    return cache_key


def cache_instances(func):
    """Decorator to handle request caching for build_all_requests"""

    @wraps(func)
    def wrapper(
        self,
        *,
        limit: Union[int, None] = None,
        samples: Optional[List[int]] = None,
        rank: int = 0,
        world_size: int = 1,
        cache_requests: bool = False,
        rewrite_requests_cache: bool = False,
        system_instruction: Optional[str] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Optional[Callable] = None,
        tokenizer_name: str = "",
        **kwargs,
    ):
        # If caching is disabled, just call the original function
        # The method will handle setting self._instances
        if not cache_requests:
            return func(
                self,
                limit=limit,
                samples=samples,
                rank=rank,
                world_size=world_size,
                cache_requests=cache_requests,
                rewrite_requests_cache=rewrite_requests_cache,
                system_instruction=system_instruction,
                apply_chat_template=apply_chat_template,
                fewshot_as_multiturn=fewshot_as_multiturn,
                chat_template=chat_template,
                tokenizer_name=tokenizer_name,
                **kwargs,
            )

        # Build cache key
        cache_key = _build_cache_key(
            self._config.task,
            self.config.num_fewshot,
            rank,
            world_size,
            apply_chat_template,
            fewshot_as_multiturn,
            system_instruction,
            tokenizer_name,
        )

        # Try to load from cache
        cached_instances = load_from_cache(file_name=cache_key, cache=cache_requests)

        # Return cached instances if available and not rewriting
        if cached_instances and not rewrite_requests_cache:
            cached_instances = (
                cached_instances[:limit] if limit is not None else cached_instances
            )
            flattened_instances = [
                instance
                for instance_group in cached_instances
                for instance in instance_group
            ]
            self._instances = flattened_instances
            eval_logger.debug(
                f"Using {len(flattened_instances)}contexts for {self.config.task} on rank {rank}..."
            )
            return

        # Store original limit for later use
        original_limit = limit

        # Process all documents when caching for simplicity
        if limit is not None:
            limit = None

        # Call the original function with modified parameters
        instances = func(
            self,
            limit=limit,
            samples=samples,
            rank=rank,
            world_size=world_size,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=chat_template,
            tokenizer_name=tokenizer_name,
            **kwargs,
        )

        # Check if method handled everything (non-cache mode returns None)
        if instances is None:
            return

        # Apply original limit if specified
        sliced_instances = (
            instances[:original_limit] if original_limit is not None else instances
        )

        # Flatten and set instances
        flattened_instances = [
            instance
            for instance_group in sliced_instances
            for instance in instance_group
        ]
        self._instances = flattened_instances

        # Validate results
        if len(self._instances) == 0:
            raise ValueError("task.build_requests() did not find any docs!")

        # Save to cache if we generated new instances
        if not cached_instances or rewrite_requests_cache:
            save_to_cache(file_name=cache_key, obj=instances)

    return wrapper
