# MIMIC-III Report Summarization

### Paper

Title: `MIMIC-III, a freely accessible critical care database`

Abstract: [https://www.nature.com/articles/sdata201635](https://www.nature.com/articles/sdata201635)

MIMIC-III containins de-identified health data from around 40,000 patients admitted to
intensive care units at a large tertiary care hospital. This task focuses on radiology
report summarization.


#### Tasks

* `mimic_repsum`: Generate extractive notes summaries, evaluated with [Radgraph-F1](https://www.cell.com/patterns/fulltext/S2666-3899(23)00157-5), bleu, rouge, bert_score, bleurt.
* `mimic_repsum_perplexity`: Generate extractive notes summaries, evaluated with perplexity.

### Citation

```bibtex
@article{johnson2016mimic,
  title={MIMIC-III, a freely accessible critical care database},
  author={Johnson, Alistair EW and Pollard, Tom J and Shen, Lu and Lehman, Li-wei H and Feng, Mengling and Ghassemi, Mohammad and Moody, Benjamin and Szolovits, Peter and Anthony Celi, Leo and Mark, Roger G},
  journal={Scientific data},
  volume={3},
  number={1},
  pages={1--9},
  year={2016},
  publisher={Nature Publishing Group}
}
```
