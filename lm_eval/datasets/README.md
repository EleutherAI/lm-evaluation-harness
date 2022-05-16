# datasets

This directory contains custom HuggingFace [dataset loading scripts](https://huggingface.co/docs/datasets/dataset_script). They are provided to maintain backward compatibility with the ad-hoc data downloaders in earlier versions of the `lm-evaluation-harness` before HuggingFace [`datasets`](https://huggingface.co/docs/datasets/index) was adopted as the default downloading manager. For example, some instances in the HuggingFace `datasets` repository process features (e.g. whitespace stripping, lower-casing, etc.) in ways that the `lm-evaluation-harness` did not.

__NOTE__: We are __not__ accepting any additional loading scripts into the main branch! If you'd like to use a custom dataset, fork the repo and follow HuggingFace's loading script guide found [here](https://huggingface.co/docs/datasets/dataset_script). You can then override your `Task`'s `DATASET_PATH` attribute to point to this script's local path.


__WARNING__: A handful of loading scripts are included in this collection because they have not yet been pushed to the Huggingface Hub or a HuggingFace organization repo. We will remove such scripts once pushed.
