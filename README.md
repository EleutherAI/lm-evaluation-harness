# CALAMITA's fork of Language Model Evaluation Harness

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10256836.svg)](https://doi.org/10.5281/zenodo.10256836) -->

![CALAMITA LOGO](/docs/img/logo_calamita.png)

[![Paper](https://img.shields.io/badge/Paper-CLiC%20IT-red)](https://clic2024.ilc.cnr.it/wp-content/uploads/2024/12/116_calamita_preface_long.pdf)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-live-yellow)](https://calamita-ailc.github.io/calamita2024/)

This repository contains the evaluation code we used in CALAMITA.
It is a fork of the lm-eval-harness. We forked at the main branch commit: **b2bf7bc4a601c643343757c92c1a51eb69caf1d7**.

## Getting started

Start by generating a **dedicated python environment** and pip-install locally the project:

```bash
git clone https://github.com/CALAMITA-AILC/CALAMITA-harness.git
cd CALAMITA-harness
pip install -e .
```

## Preparing the data

Most of CALAMITA tasks run on publicly available data. Such datasets will be automatically downloaded when you will launch the runner script for the first time.
However, a small number of tasks (see below) requires using private data. As of now, you will need to obtain the data from the authors, and copy it locally. Please follow these steps:

1. Create a folder in the main project directory called `private-data` 
2. Add the following directories and files:
    * blm-IT
        * blm-IT/task_data/BLM-AgrIt_type-II/train.json 
        * blm-IT/task_data/BLM-AgrIt_type-II/test.json
    * verifIT
        * veryfIT_small.json
        * veryfIT_full.json
    * Multi-IT
        * Multi-IT/Multi-IT-C/quiz.json


## Cite as

```bibtex
@inproceedings{calamita2024,
    title={{CALAMITA – Challenge the Abilities of LAnguage Models in ITAlian: Overview}},
    author={Attanasio, Giuseppe and Basile, Pierpaolo and Borazio, Federico and Croce, Danilo and Francis, Maria and Gili, Jacopo and Musacchio, Elio and Nissim, Malvina and Patti, Viviana and Rinaldi, Matteo and Scalena, Daniel},
    booktitle={Proceedings of the 10th Italian Conference on Computational Linguistics (CLiC-it 2024)},
    year={2024}
}
```
