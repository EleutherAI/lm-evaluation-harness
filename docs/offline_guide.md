# Offline Guide

`lm-evaluation-harness` can be used on a machine without internet access by first downloading the evaluation datasets and then running the evaluation script:

1. On a machine with internet access, download the datasets using the `scripts/save_local.py`.
```
python scripts/save_local.py --tasks <list of tasks as in the lm_eval commands> --local_base_dir <local directory to store datasets>
```

If the `--local_base_dir` argument is not provided, the datasets will be downloaded to the default directory `~/.lm_eval/datasets`.
It is also possible to specify a custom directory through an environment variable `LM_EVAL_DATASETS_PATH`.

2. Transfer the downloaded datasets to the machine without internet access. This step may not be required if the machine without internet access has access to the datasets through a shared file system.

3. Run the evaluation script on the machine without internet access.
```
lm_eval ... --tasks <tasks to run> --load_local --local_base_dir <local_base_dir> ...
```

Similar to above, if the `--local_base_dir` argument is not provided, the datasets will be loaded from the default directory `~/.lm_eval/datasets`, or from the directory specified by the environment variable `LM_EVAL_DATASETS_PATH`.
