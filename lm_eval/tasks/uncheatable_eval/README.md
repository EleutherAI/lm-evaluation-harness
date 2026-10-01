# Uncheatable Eval

These tasks evaluate autoregressive language models on [Uncheatable Eval](https://github.com/Jellyfish042/uncheatable_eval). Each task measures rolling log-likelihood over recently-published documents across Wikipedia, GitHub, BBC News, arXiv, bioRxiv, and AO3.

### Citation

```text
@software{uncheatable_eval,
  author       = {Jellyfish042},
  title        = {Uncheatable Eval},
  month        = may,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.11284692},
  url          = {https://zenodo.org/record/11284692}
}
```

### Groups and Tasks

#### Groups

* `uncheatable_eval`: aggregates every category in the snapshot, weighted by size.

#### Tasks

* `uncheatable_eval_wikipedia_english`
* `uncheatable_eval_wikipedia_nonenglish`
* `uncheatable_eval_github_python`
* `uncheatable_eval_github_cpp`
* `uncheatable_eval_github_javascript`
* `uncheatable_eval_github_markdown`
* `uncheatable_eval_github_other`
* `uncheatable_eval_bbc_news`
* `uncheatable_eval_arxiv_physics`
* `uncheatable_eval_arxiv_math`
* `uncheatable_eval_arxiv_cs`
* `uncheatable_eval_arxiv_other`
* `uncheatable_eval_biorxiv_all`
* `uncheatable_eval_ao3_english`
* `uncheatable_eval_ao3_nonenglish`

### Dataset

Documents come from [`Jellyfish042/UncheatableEval-2026-07`](https://huggingface.co/datasets/Jellyfish042/UncheatableEval-2026-07),
a single `test` split whose `category` column carries the domain. Each task selects one
category and scores its `content` field, matching upstream's `evaluator.py`.

New snapshots are published monthly to the
[UncheatableEval collection](https://huggingface.co/collections/Jellyfish042/uncheatableeval).
To evaluate against a fresher one, bump `dataset_path` in `_uncheatable_eval_base.yaml`.
Numbers are only comparable across models evaluated on the *same* snapshot.

### Changelog
