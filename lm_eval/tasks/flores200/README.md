# FLORES-200

### Paper

Title: `The FLORES-200 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation`
Link: https://github.com/facebookresearch/flores/blob/main/flores200/README.md

Original Flores-101 paper: https://arxiv.org/abs/2106.03193

The creation of FLORES-200 doubles the existing language coverage of FLORES-101. Given the nature of the new languages, which have less standardization and require more specialized professional translations, the verification process became more complex. This required modifications to the translation workflow. FLORES-200 has several languages which were not translated from English. Specifically, several languages were translated from Spanish, French, Russian and Modern Standard Arabic. Moreover, FLORES-200 also includes two script alternatives for four languages.

Homepage: https://github.com/facebookresearch/flores/tree/main/flores200

We use the prompt template introduced by "Multilingual Machine Translation with Large Language Models:
Empirical Results and Analysis" https://arxiv.org/pdf/2304.04675.pdf, and then further used in "SambaLingo: Teaching Large Language Models New Languages" https://arxiv.org/abs/2404.05829.

### Citation

```
@article{nllb2022,
  author    = {NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi,  Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Jeff Wang},
  title     = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  year      = {2022}
}

@inproceedings{,
  title={The FLORES-101  Evaluation Benchmark for Low-Resource and Multilingual Machine Translation},
  author={Goyal, Naman and Gao, Cynthia and Chaudhary, Vishrav and Chen, Peng-Jen and Wenzek, Guillaume and Ju, Da and Krishnan, Sanjana and Ranzato, Marc'Aurelio and Guzm\'{a}n, Francisco and Fan, Angela},
  year={2021}
}

@inproceedings{,
  title={Two New Evaluation Datasets for Low-Resource Machine Translation: Nepali-English and Sinhala-English},
  author={Guzm\'{a}n, Francisco and Chen, Peng-Jen and Ott, Myle and Pino, Juan and Lample, Guillaume and Koehn, Philipp and Chaudhary, Vishrav and Ranzato, Marc'Aurelio},
  journal={arXiv preprint arXiv:1902.01382},
  year={2019}
}
```

### Groups and Tasks

#### Tasks
There are 41618 supported translation tasks. In order to find the task name use the following steps.

- find the language code and character set you are interested in by visiting https://github.com/facebookresearch/flores/tree/main/flores200. The 3 characters before the underscore are the 3 letter [ISO 639](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) language codes, the 4 characters after the underscore represent the [ISO 15924](https://en.wikipedia.org/wiki/ISO_15924) codes for the representation of names of scripts. (For English your LANG_CODE would "eng_Latn")
- search for language code in task directory `cd lm_eval/tasks/flores200 && ls | grep LANG_CODE`. If you want to search for a specific translation task, you can grep for two distinct language codes, for example for Arabic -> English translation you could search `cd lm_eval/tasks/flores200 && ls | grep arb_Arab | grep eng_Latn`

### Checklist

For adding novel benchmarks/datasets to the library:
  * [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
