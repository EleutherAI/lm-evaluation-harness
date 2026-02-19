import re
import signal
from typing import Optional

# Re-use the hendrycks_math utils
from lm_eval.tasks.hendrycks_math.utils import process_docs, process_results  # noqa: F401