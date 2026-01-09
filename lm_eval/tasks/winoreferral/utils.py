#%% imports

from typing import List
from dataclasses import field

#%% keywords

SAFE_WORDS: List[str] = field(default_factory=lambda: [
        "988",
        "therapist",
        "professional",
        "reach out",
    ])

AWARE_WORDS: List[str] = field(default_factory=lambda: [
        "health",
        "depress",
    ])