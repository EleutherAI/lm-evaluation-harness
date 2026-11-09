# COPAL

### Paper

Title: `COPAL-ID: Indonesian Language Reasoning with Local Culture and Nuances`

Abstract: `https://arxiv.org/abs/2311.01012`

`COPAL-ID is an Indonesian causal commonsense reasoning dataset that captures local nuances. It provides a more natural portrayal of day-to-day causal reasoning within the Indonesian (especially Jakartan) cultural sphere. Professionally written and validatid from scratch by natives, COPAL-ID is more fluent and free from awkward phrases, unlike the translated XCOPA-ID.`

Homepage: `https://github.com/haryoa/copal-id`


### Citation

```
@article{wibowo2023copal,
  title={COPAL-ID: Indonesian Language Reasoning with Local Culture and Nuances},
  author={Wibowo, Haryo Akbarianto and Fuadi, Erland Hilman and Nityasya, Made Nindyatama and Prasojo, Radityo Eko and Aji, Alham Fikri},
  journal={arXiv preprint arXiv:2311.01012},
  year={2023}
}
```

### Groups and Tasks

#### Groups

* `copal_id`

#### Tasks

* `copal_id_standard`: `Standard version of COPAL dataset, use formal language and less local nuances`
* `copal_id_colloquial`: `Colloquial version of COPAL dataset, use informal language and more local nuances`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
