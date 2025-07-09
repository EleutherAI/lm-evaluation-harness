# Permutation Composition Benchmark

This suite of tasks evaluates a model's ability to perform permutation composition, a key measure of its state-tracking and symbolic reasoning capabilities. The theoretical basis for this benchmark is presented in the paper "The Illusion of State in State-Space Models."

The tasks use the `BeeGass/permutation-groups` dataset on the Hugging Face Hub.

### Task Validity Checklist

- [x] Is the task an existing benchmark in the literature?
  - [x] Have you referenced the original paper that introduced the task?
  - [x] The original paper provides the theoretical framework. The `BeeGass/permutation-groups` dataset is a new, concrete implementation of this concept. Its correctness can be verified independently via the `examples.py` script in its [source repository](https://github.com/BeeGass/permutation-groups).