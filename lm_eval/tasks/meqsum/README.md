# MeqSum

### Paper

Title: `On the Summarization of Consumer Health Questions`

Abstract: [https://aclanthology.org/P19-1215/](https://aclanthology.org/P19-1215/)

Question understanding is one of the main challenges in question answering. In real world
applications, users often submit natural language questions that are longer than needed
and include peripheral information that increases the complexity of the question, leading
to substantially more false positives in answer retrieval. In this paper, we study neural
abstractive models for medical question summarization. We introduce the MeQSum corpus of
1,000 summarized consumer health questions.


### Dataset loading

The task reads the publisher's `meqsum_source` Parquet export at revision
`3d16738df5528f99adc4bb28a9dd78bc99d5433b`, rather than the dataset loading
script, which is unsupported by `datasets>=4`. The explicit `data_files` entry
selects the source schema instead of mixing it with the BigBio text-to-text export.

All 1,000 records, their order, and the rendered prompts and reference summaries
match the original MeQSum spreadsheet. The existing split selection, preprocessing,
metrics, and task version 1.0 are retained.

### Citation

```bibtex
@inproceedings{ben-abacha-demner-fushman-2019-summarization,
    title = "On the Summarization of Consumer Health Questions",
    author = "Ben Abacha, Asma  and
      Demner-Fushman, Dina",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1215",
    doi = "10.18653/v1/P19-1215",
    pages = "2228--2234"}
```
