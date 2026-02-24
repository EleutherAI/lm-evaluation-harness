from ._task import Task  # noqa need to import Task first to avoid circular imports
from ._generation import GenerateTask
from ._ll import LoglikelihoodRollingTask, LoglikelihoodTask, MultipleChoiceTask


Task._registry = {
    "multiple_choice": MultipleChoiceTask,
    "generate_until": GenerateTask,
    "loglikelihood": LoglikelihoodTask,
    "loglikelihood_rolling": LoglikelihoodRollingTask,
}

# Backward compatibility alias
ConfigurableTask = Task


__all__ = [
    "ConfigurableTask",
    "LoglikelihoodRollingTask",
    "LoglikelihoodTask",
    "MultipleChoiceTask",
    "Task",
]
