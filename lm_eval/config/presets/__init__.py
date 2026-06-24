# Import subclasses to trigger registration via __init_subclass__
from .generation import COTGeneratePreset, GeneratePreset
from .multiple_choice import ClozePreset, MCQPreset
from .preset import PresetConfig
