# Polish PSC

### Description

The Polish Summaries Corpus (PSC) is a dataset of summaries for 569 news articles. The human annotators created five extractive summaries for each article by choosing approximately 5% of the original text. A different annotator created each summary. The subset of 154 articles was also supplemented with additional five abstractive summaries each, i.e., not created from the fragments of the original article. In huggingface version of this dataset, summaries of the same article are used as positive pairs, and the most similar summaries of different articles are sampled as negatives.


### Citation

```
@inproceedings{ogro:kop:14:lrec,
  title={The {P}olish {S}ummaries {C}orpus},
  author={Ogrodniczuk, Maciej and Kope{'c}, Mateusz},
  booktitle = "Proceedings of the Ninth International {C}onference on {L}anguage {R}esources and {E}valuation, {LREC}~2014",
  year = "2014",
}
```
