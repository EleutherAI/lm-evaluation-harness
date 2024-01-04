# BBQ

### Paper

Title: BBQ: A Hand-Built Bias Benchmark for Question Answering

Abstract: https://aclanthology.org/2022.findings-acl.165/

BBQ measures the bias in the output for the question answering task.
The dataset of question-sets constructed by the authors that highlight attested social biases against people belonging to protected classes along nine social dimensions relevant for U.S. English-speaking contexts.
BBQ evaluates model responses at two levels: (i) given an under-informative context, how strongly responses reflect social biases (AMBIGUOUS CONTEXT), and (ii) given an adequately informative context, whether the model's biases override a correct answer choice (DISAMBIGUATED CONTEXT).

Homepage: https://github.com/nyu-mll/BBQ


### Citation

```
@inproceedings{parrish-etal-2022-bbq,
    title = "{BBQ}: A hand-built bias benchmark for question answering",
    author = "Parrish, Alicia  and
      Chen, Angelica  and
      Nangia, Nikita  and
      Padmakumar, Vishakh  and
      Phang, Jason  and
      Thompson, Jana  and
      Htut, Phu Mon  and
      Bowman, Samuel",
    editor = "Muresan, Smaranda  and
      Nakov, Preslav  and
      Villavicencio, Aline",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.165",
    doi = "10.18653/v1/2022.findings-acl.165",
    pages = "2086--2105"
}
```

### Groups and Tasks

#### Groups

* `bbq`: Tests the bias for all categories in the ambiguous and disambiguated contexts.

#### Tasks
The following tasks evaluate the accuracy on BBQ for the different categories of bias:
* `bbq_age`: Age
* `bbq_disability`: Disability status
* `bbq_gender`: Gender
* `bbq_nationality`: Nationality
* `bbq_physical_appearance`: Physical appearance
* `bbq_race_ethnicity`: Race/ethnicity
* `bbq_religion`: Religion
* `bbq_ses`: Socio-economic status
* `bbq_sexual_orientation`: Sexual orientation

Two intersectional bias categories exist as well:
* `bbq_race_x_gender`: The intersection of race/ethnicity and gender
* `bbq_race_x_ses`: The intersection of race/ethnicity and socio-economic status
However, this is in the current implementation not really taken into account in computing the bias scores.

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
