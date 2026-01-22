# IFBench

### Paper

Title: Generalizing Verifiable Instruction Following  
Abstract: https://arxiv.org/pdf/2507.02833

IFBench is a benchmark for precise instruction following with new out-of-distribution constraints and verification functions.

Homepage: https://github.com/allenai/IFBench

### Citation

```
@misc{pyatkin2025generalizing,
  title={Generalizing Verifiable Instruction Following},
  author={Valentina Pyatkin and Saumya Malik and Victoria Graf and Hamish Ivison and Shengyi Huang and Pradeep Dasigi and Nathan Lambert and Hannaneh Hajishirzi},
  year={2025},
  journal={Advances in Neural Information Processing Systems},
  volume={38}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `ifbench`: OOD instruction-following benchmark with verifier-based metrics.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
