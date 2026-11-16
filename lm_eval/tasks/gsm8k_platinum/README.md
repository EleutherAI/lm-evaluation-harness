# GSM8k Platinum

GSM8K Platinum is a revised version of the full test set of GSM8K (Grade School Math 8K), a dataset of grade school math word problems. To revise this dataset, we ran a variety of frontier models each individual example and manually re-annotated any example for which at least one model made an error. We revise the labels of mislabeled examples, and remove any question that we determine to be poorly written (most often due to ambiguity in the problem statement). See our paper for further details on the revision process and our criteria for "bad" questions.

## Paper
Do Large Language Model Benchmarks Test Reliability?
https://arxiv.org/abs/2502.03461

NOTE: See the official implementation of the task:
    https://github.com/MadryLab/platinum-benchmarks/
for how to make use of the dataset's calculator annotations in your language
model's sample/generation function.

Blog: https://gradientscience.org/gsm8k-platinum/
Homepage: http://platinum-bench.csail.mit.edu/


## Citation
```
@misc{vendrow2025largelanguagemodelbenchmarks,
      title={Do Large Language Model Benchmarks Test Reliability?},
      author={Joshua Vendrow and Edward Vendrow and Sara Beery and Aleksander Madry},
      year={2025},
      eprint={2502.03461},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.03461},
}

@misc{cobbe2021training,
      title={Training Verifiers to Solve Math Word Problems},
      author={Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
      year={2021},
      eprint={2110.14168},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Groups and Tasks

#### Groups

- `math_word_problems`
- `chain_of_thought`
- `self_consistency`

#### Tasks

- `gsm8k_platinum`
- `gsm8k_platinum_cot`: GSM8K Platinum with Chain-of-Thought
- `gsm8k_platinum_cot_self_consistency`: GSM8K Platinum with Chain-of-Thought and Self-Consistency
- `gsm8k_platinum_cot_llama`: GSM8K Platinum with prompt formatting modified to conform to the evaluation settings described by Meta here: https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-8B-Instruct-evals/viewer/Meta-Llama-3.1-8B-Instruct-evals__gsm8k__details?row=0
    - Use this task with --fewshot_as_multiturn and --apply_chat_template to replicate Meta's reported performance.
