# Import subclasses to trigger registration via __init_subclass__
from ._base import FormatConfig
from ._generation import COTGeneratePreset, GenerateFormat
from ._multiple_choice import ClozePreset, MCQAFormat


__all__ = [
    "COTGeneratePreset",
    "ClozePreset",
    "GenerateFormat",
    "MCQAFormat",
    "FormatConfig",
]
