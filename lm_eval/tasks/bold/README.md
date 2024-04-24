# BOLD

### Paper

Title: `Bias in Open-ended Language Generation Dataset (BOLD)`

Abstract: https://arxiv.org/abs/2101.11718

Bias in Open-ended Language Generation Dataset (BOLD) is a dataset to evaluate fairness in open-ended language generation in English language. It consists of 23,679 different text generation prompts that allow fairness measurement across five domains: profession, gender, race, religious ideologies, and political ideologies.

Homepage: https://github.com/amazon-research/bold

### Citation

```
@inproceedings{bold_2021,
  author = {Dhamala, Jwala and Sun, Tony and Kumar, Varun and Krishna, Satyapriya and Pruksachatkun, Yada and Chang, Kai-Wei and Gupta, Rahul},
  title = {BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation},
  year = {2021},
  isbn = {9781450383097},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3442188.3445924},
  doi = {10.1145/3442188.3445924},
  booktitle = {Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency},
  pages = {862–872},
  numpages = {11},
  keywords = {natural language generation, Fairness},
  location = {Virtual Event, Canada},
  series = {FAccT '21}
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
