# QuAC

### Paper

Title: `QuAC: Question Answering in Context`

Abstract: https://arxiv.org/abs/1808.07036

QuAC evaluates information-seeking question answering in a dialogue. A student
asks a sequence of questions about a hidden Wikipedia section, and a teacher
answers with spans from that section or marks a question as unanswerable.

Homepage: https://quac.ai/

This task follows the HELM QuAC scenario. Each example contains the article
title, background, section, passage, and preceding dialogue turns. HELM uses a
fixed random seed to select one question from each dialogue and scores the
answer with the maximum token F1 over all references. The task uses the
script-free `lighteval/quac_helm` mirror of those prompts because the original
`allenai/quac` dataset requires a loading script that is unsupported by current
versions of `datasets`.

Reference implementations and data:

* HELM scenario: https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/quac_scenario.py
* Script-free dataset: https://huggingface.co/datasets/lighteval/quac_helm
* Original dataset: https://huggingface.co/datasets/allenai/quac

### Validation

Validate the task configuration and run a small evaluation with:

```bash
lm-eval validate --tasks quac
lm-eval run --model hf --model_args pretrained=EleutherAI/pythia-14m,dtype=float32 --tasks quac --limit 10 --batch_size 1 --device cpu
```

### Citation

```bibtex
@inproceedings{choi-etal-2018-quac,
    title = "{Q}u{AC}: Question Answering in Context",
    author = "Choi, Eunsol  and He, He  and Iyyer, Mohit  and Yatskar, Mark  and Yih, Wen-tau  and Choi, Yejin  and Liang, Percy  and Zettlemoyer, Luke",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1241/",
    doi = "10.18653/v1/D18-1241",
    pages = "2174--2184",
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `quac`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
