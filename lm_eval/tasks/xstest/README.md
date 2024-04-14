# XSTest

### Paper

Title: `XSTest Dataset for Testing Exaggerated Safety`

Abstract: https://arxiv.org/abs/2307.04657

Without proper safeguards, large language models will follow malicious instructions and generate toxic content. This motivates safety efforts such as red-teaming and large-scale feedback learning, which aim to make models both helpful and harmless. However, there is a tension between these two objectives, since harmlessness requires models to refuse to comply with unsafe prompts, and thus not be helpful. Recent anecdotal evidence suggests that some models may have struck a poor balance, so that even clearly safe prompts are refused if they use similar language to unsafe prompts or mention sensitive topics. With XSTest, we introduce a new test suite to identify such eXaggerated Safety behaviours in a structured and systematic way. XSTest comprises 250 safe prompts across ten prompt types that well-calibrated models should not refuse to comply with, and 200 unsafe prompts as contrasts that, for most LLM applications, should be refused. In our paper, we describe XSTest's creation and composition, and use the test suite to highlight systematic failure modes in state-of-the-art language models as well as more general challenges in building safer language models.

Homepage: https://github.com/paul-rottger/exaggerated-safety/
### Citation

```
@misc{röttger2023xstest,
      title={XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models}, 
      author={Paul Röttger and Hannah Rose Kirk and Bertie Vidgen and Giuseppe Attanasio and Federico Bianchi and Dirk Hovy},
      year={2023},
      eprint={2308.01263},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
