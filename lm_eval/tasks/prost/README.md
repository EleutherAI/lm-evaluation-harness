# PROST

### Paper

Title: `PROST: Physical Reasoning about Objects Through Space and Time`

Abstract: https://arxiv.org/abs/2106.03634

PROST, Physical Reasoning about Objects Through Space and Time, is a dataset
consisting of 18,736 multiple-choice questions made from 14 manually curated
templates, covering 10 physical reasoning concepts. All questions are designed
to probe both causal and masked language models in a zero-shot setting.

NOTE: PROST is limited to the zero-shot setting to adhere to authors' intentions
as discussed in section 7 of the paper: "We hope that the community will use
this dataset in the intended way: in a zero-shot setting to probe models which
have been trained on data not specifically collected to succeed on PROST."

Homepage: https://github.com/nala-cub/prost


### Citation

```
@inproceedings{aroca-ouellette-etal-2021-prost,
    title = "{PROST}: {P}hysical Reasoning about Objects through Space and Time",
    author = "Aroca-Ouellette, St{\'e}phane  and
      Paik, Cory  and
      Roncone, Alessandro  and
      Kann, Katharina",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.404",
    pages = "4597--4608",
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet.

#### Tasks

* `prost`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
