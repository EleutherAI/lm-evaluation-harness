import ipaddress
import json
import logging
import os
import signal
import uuid
from typing import Any, Dict


# check if jsonschema is installed
try:
    import jsonschema
    from jsonschema import Draft202012Validator, FormatChecker, ValidationError
except ImportError as e:
    raise ImportError(
        "jsonschema is not installed. Please install it using 'pip install jsonschema[format]'"
    ) from e

eval_logger = logging.getLogger(__name__)


class ValidationTimeout(Exception):
    """Raised when schema validation exceeds the per-sample time budget."""


# Per-sample validation time budget (seconds). The original JSONSchemaBench
# treats a schema that cannot be processed within a fixed budget as a
# processing failure; without such a cap a single pathological schema/instance
# (e.g. deeply-recursive ``$ref`` or a format regex with catastrophic
# backtracking) can spin the CPU indefinitely and hang the whole evaluation.
# Override with the ``JSONSCHEMA_BENCH_TIMEOUT_S`` environment variable.
VALIDATION_TIMEOUT_S = int(os.environ.get("JSONSCHEMA_BENCH_TIMEOUT_S", "40"))


def _run_with_timeout(fn, seconds: int = VALIDATION_TIMEOUT_S):
    """Run ``fn()`` under a wall-clock ``SIGALRM`` timeout.

    Raises :class:`ValidationTimeout` if ``fn`` runs longer than ``seconds``.
    If ``SIGALRM`` is unavailable (non-main thread, or a platform without it),
    ``fn`` is run without a timeout so behavior never regresses to a crash.
    """

    def _handler(signum, frame):
        raise ValidationTimeout(f"validation exceeded {seconds}s")

    try:
        old_handler = signal.signal(signal.SIGALRM, _handler)
    except (ValueError, AttributeError):
        # Not in the main thread, or SIGALRM not supported: run unguarded.
        return fn()

    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def is_json_schema_valid(schema: dict):
    """
    Check if a JSON schema is valid.

    :param schema: A JSON schema.
    :return: True if the schema is valid, False otherwise.
    """
    try:
        # Check if the schema is valid
        jsonschema.Draft202012Validator.check_schema(schema)
        return True
    except jsonschema.SchemaError:
        return False


# Initialize the FormatChecker
format_checker = FormatChecker()


# Add custom format checkers
@format_checker.checks("ipv4")
def ipv4_check(value):
    ipaddress.IPv4Address(value)


@format_checker.checks("ipv6")
def ipv6_check(value):
    ipaddress.IPv6Address(value)


@format_checker.checks("uuid")
def uuid_check(value):
    uuid.UUID(value)


def schema_conform_with_format_checker(
    instance: Dict[str, Any], schema: Dict[str, Any]
) -> bool:
    """
    Validate a JSON instance against a schema with enhanced format checking.

    :param schema: The JSON schema to validate against.
    :param instance: The JSON instance to validate.
    :raises ValidationError: If the validation fails.
    """
    # first check if the schema is valid
    if not is_json_schema_valid(schema):
        raise ValidationError("The JSON schema is invalid.")
    validator = Draft202012Validator(schema, format_checker=format_checker)
    try:
        validator.validate(instance)
    except ValidationError as e:
        raise ValidationError(e.message)
    return True


def schema_compliance(references: list[str], predictions: list[str]) -> bool:
    assert len(references) == 1, (
        "We only have one reference for this task, which is the JSON schema."
    )
    assert len(predictions) == 1, (
        "Currently, we don't support pass@k for JSON schema validation."
    )
    reference = references[0]
    prediction = predictions[0]  # Since predictions is a list of lists

    json_schema = json.loads(reference.strip())
    try:
        json_obj = json.loads(prediction.strip().strip("```").strip("json"))
    except json.JSONDecodeError:
        return False

    try:
        schema_conform = _run_with_timeout(
            lambda: schema_conform_with_format_checker(json_obj, json_schema)
        )
    except ValidationTimeout as e:
        # Treat an un-processable schema/instance as non-compliant instead of
        # hanging the evaluation forever.
        eval_logger.error(f"Error: {e}; treating as non-compliant")
        return False
    except Exception as e:
        eval_logger.error(f"Error: {e}")
        return False

    return schema_conform


def json_validity(references: list[str], predictions: list[str]) -> bool:
    assert len(predictions) == 1, (
        "Currently, we don't support pass@k for JSON schema validation."
    )
    prediction = predictions[0]  # Since predictions is a list of lists
    try:
        json.loads(prediction.strip().strip("```").strip("json").strip())
    except json.JSONDecodeError:
        return False
    return True
