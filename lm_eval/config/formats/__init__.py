# Import subclasses to trigger registration via __init_subclass__
from ._base import FormatConfig
from ._generation import COTGenFormat, GenerateFormat
from ._multiple_choice import ClozeFormat, MCQAFormat


__all__ = [
    "COTGenFormat",
    "ClozeFormat",
    "FormatConfig",
    "GenerateFormat",
    "MCQAFormat",
]
