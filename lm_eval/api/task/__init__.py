from ._generation import GenerateTask
from ._ll import LoglikelihoodRollingTask, LoglikelihoodTask, MultipleChoiceTask
from ._task import Task


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
