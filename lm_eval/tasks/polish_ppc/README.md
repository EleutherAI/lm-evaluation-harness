# Polish PPC

### Description

Polish Paraphrase Corpus contains 7000 manually labeled sentence pairs. The dataset was divided into training, validation and test splits. The training part includes 5000 examples, while the other parts contain 1000 examples each. The main purpose of creating such a dataset was to verify how machine learning models perform in the challenging problem of paraphrase identification, where most records contain semantically overlapping parts. Technically, this is a three-class classification task, where each record can be assigned to one of the following categories:
- Exact paraphrases - Sentence pairs that convey exactly the same information. We are interested only in the semantic meaning of the sentence, therefore this category also includes sentences that are semantically identical but, for example, have different emotional emphasis.
- Close paraphrases - Sentence pairs with similar semantic meaning. In this category we include all pairs which contain the same information, but in addition to it there may be other semantically non-overlapping parts. This category also contains context-dependent paraphrases - sentence pairs that may have the same meaning in some contexts but are different in others.
- Non-paraphrases - All other cases, including contradictory sentences and semantically unrelated sentences.

The corpus contains 2911, 1297, and 2792 examples for the above three categories, respectively. The process of annotating the dataset was preceded by an automated generation of candidate pairs, which were then manually labeled. We experimented with two popular techniques of generating possible paraphrases: backtranslation with a set of neural machine translation models and paraphrase mining using a pre-trained multilingual sentence encoder. The extracted sentence pairs are drawn from different data sources: Taboeba, Polish news articles, Wikipedia and Polish version of SICK dataset. Since most of the sentence pairs obtained in this way fell into the first two categories, in order to balance the dataset, some of the examples were manually modified to convey different information. In this way, even negative examples often have high semantic overlap, making this problem difficult for machine learning models.


### Citation

```
@inproceedings{9945218,
  author={Dadas, S{\l}awomir},
  booktitle={2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC)}, 
  title={Training Effective Neural Sentence Encoders from Automatically Mined Paraphrases}, 
  year={2022},
  volume={},
  number={},
  pages={371-378},
  doi={10.1109/SMC53654.2022.9945218}
}
```
