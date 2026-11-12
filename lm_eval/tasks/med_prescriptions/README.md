# Medical prescription selection

`med_prescriptions_easy` and `med_prescriptions_hard` are multiple-choice tasks
using the `devlocalhost/prescription-full` dataset. They select a gold prescription
from each retained record and place it among three sampled alternatives.

## Version 1.2

Both sets of alternatives are sampled once during `process_docs` and saved in the
processed record. Prompt rendering and log-likelihood request construction read
these saved choices, so the model is scored against exactly the alternatives
displayed in its prompt. Repeated rendering and reloading a processed dataset
preserve the same choices.

Version 1.1 sampled alternatives separately each time `doc_to_choice` was called.
The prompt and scored continuations could therefore contain different negative
options. Version 1.2 changes the sampling order and evaluation inputs; its scores
should not be compared directly with results from version 1.1. The hard task
inherits this version from the easy task configuration.
