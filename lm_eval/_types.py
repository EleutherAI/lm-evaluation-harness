from __future__ import annotations

from typing import Literal


# multiple_choice types send a number of "loglikelihood" instances
OutputType = Literal["loglikelihood", "loglikelihood_rolling", "generate_until"]
