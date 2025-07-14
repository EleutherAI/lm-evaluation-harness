# Dynamic IFEval

## Task Summary
This benchmark evaluates an LLM's ability to follow verifiable instructions that are grounded in specific input texts.

### Overview
- A dataset of natural language texts is used as the source of truth.
- A predefined set of instruction types defines the structure of rules (e.g., "The first letter is", "The number of words is", etc.).
- For each instance, a subset of instruction types is sampled, and a text is randomly selected from the dataset.
- Based on these, a set of instructions is generated such that the original text satisfies all the given constraints.
- The model is prompted to generate a response (typically a text) that adheres to the same instructions.

### Purpose
The benchmark ensures that:

- All instruction sets are satisfiable: they are derived from a real text that serves as a witness solution.
- Evaluation is grounded: each response can be automatically or semi-automatically verified for correctness.
- Generalization is tested: models must parse, interpret, and apply multiple constraints simultaneously.

### References
The benchmark is inspired and offers a dynamic version of
@misc{zhou2023instructionfollowingevaluationlargelanguage,
      title={Instruction-Following Evaluation for Large Language Models}, 
      author={Jeffrey Zhou and Tianjian Lu and Swaroop Mishra and Siddhartha Brahma and Sujoy Basu and Yi Luan and Denny Zhou and Le Hou},
      year={2023},
      eprint={2311.07911},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2311.07911}, 
}


### Task Validity Checklist

- [ ] Is the task an existing benchmark in the literature?
  - [ ] Have you referenced the original paper that introduced the task?
  - [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

- [ ] Is the "Main" variant of this task clearly denoted?
- [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
- [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
