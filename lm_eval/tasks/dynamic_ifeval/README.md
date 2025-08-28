# Dynamic IFEval

## Task Summary
This benchmark evaluates an LLM's ability to follow verifiable instructions that are grounded in specific input texts.

### Overview
- A dataset of natural language texts is used as the source of truth.
- A predefined set of instruction types defines the structure of rules (e.g., "The nth letter is", "The number of words is", etc.).
- For each instance, a subset of instruction types is sampled, and a text is randomly selected from the dataset.
- Based on these, a set of instructions is generated such that the original text satisfies all the given constraints.
- The model is prompted to generate a response (a text) that adheres to the same instructions.

### Purpose
The benchmark ensures that:

- All instruction sets are satisfiable: they are derived from a real text that serves as a "witness solution".
- Evaluation is grounded: each response can be automatically verified for correctness exactly, without the need for a neural reward model.

### References
The benchmark is inspired and offers a dynamic version of
```bibtex
@misc{zhou2023instructionfollowingevaluationlargelanguage,
      title={Instruction-Following Evaluation for Large Language Models}, 
      author={Jeffrey Zhou and Tianjian Lu and Swaroop Mishra and Siddhartha Brahma and Sujoy Basu and Yi Luan and Denny Zhou and Le Hou},
      year={2023},
      eprint={2311.07911},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2311.07911}, 
}
```

### Additional Requirements
datasets

### Example Usage
lm_eval \
  --model vllm \
  --model_args '{"pretrained":"Qwen/Qwen3-4B","trust_remote_code":true,"max_model_len":16384,"enable_thinking":true}' \
  --gen_kwargs '{"max_gen_toks":8192,"until":["Input","Input:","<eot_id>","<|im_end|>","###","Question","question","####","Problem","Response"],"temperature":0.6}' \
  --apply_chat_template \
  --tasks dynamic_ifeval \
  --batch_size auto \
  --output_path 'results.jsonl'

You can set the seed used for the generation of the dataset with the following flag:
  --metadata '{"dataset_seed":131}' \

### Task Validity Checklist

- [ ] Is the task an existing benchmark in the literature?
  - [ ] Have you referenced the original paper that introduced the task?
  - [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

- [ ] Is the "Main" variant of this task clearly denoted?
- [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
- [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
