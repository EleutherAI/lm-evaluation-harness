# Japanese Leaderboard

The Japanese LLM Leaderboard evaluates language models based on a wide range of NLP tasks that reflect the characteristics of the Japanese language.

### Groups, Tags, and Tasks

#### Groups

* `japanese_leaderboard`: runs all tasks defined in this directory

#### Tasks

##### Generation Evaluation

* `ja_leaderboard_jaqket_v2`: The JAQKET dataset is designed for Japanese question answering research, featuring quiz-like questions with answers derived from Wikipedia article titles. [Source](https://github.com/kumapo/JAQKET-dataset)
* `ja_leaderboard_mgsm`: Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems, proposed in the paper Language models are multilingual chain-of-thought reasoners. [Source](https://huggingface.co/datasets/juletxara/mgsm)
* `ja_leaderboard_xlsum`: This is the filtered Japanese subset of XL-Sum. [Source](https://github.com/csebuetnlp/xl-sum)
* `ja_leaderboard_jsquad`: JSQuAD is a Japanese version of SQuAD, a reading comprehension dataset. Each instance in the dataset consists of a question regarding a given context (Wikipedia article) and its answer. JSQuAD is based on SQuAD 1.1 (there are no unanswerable questions). [Source](https://github.com/yahoojapan/JGLUE)

##### Multi-Choice/Classification Evaluation

* `ja_leaderboard_jcommonsenseqa`: JCommonsenseQA is a Japanese version of CommonsenseQA, which is a multiple-choice question answering dataset that requires commonsense reasoning ability. [Source](https://github.com/yahoojapan/JGLUE)
* `ja_leaderboard_jnli`: JNLI is a Japanese version of the NLI (Natural Language Inference) dataset. The inference relations are entailment (含意), contradiction (矛盾), and neutral (中立). [Source](https://github.com/yahoojapan/JGLUE)
* `ja_leaderboard_marc_ja`: MARC-ja is a text classification dataset based on the Japanese portion of Multilingual Amazon Reviews Corpus (MARC). [Source](https://github.com/yahoojapan/JGLUE)
* `ja_leaderboard_xwinograd`: This is the Japanese portion of XWinograd. [Source](https://huggingface.co/datasets/polm-stability/xwinograd-ja)

### Citation

```bibtex
@inproceedings{ja_leaderboard_jaqket_v2,
  title         = {JAQKET: クイズを題材にした日本語 QA データセットの構築},
  author        = {鈴木正敏 and 鈴木潤 and 松田耕史 and ⻄田京介 and 井之上直也},
  year          = 2020,
  booktitle     = {言語処理学会第26回年次大会},
  url           = {https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf}
}

@article{ja_leaderboard_mgsm_1,
  title         = {Training Verifiers to Solve Math Word Problems},
  author        = {
    Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and
    Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro
    and Hesse, Christopher and Schulman, John
  },
  year          = 2021,
  journal       = {arXiv preprint arXiv:2110.14168}
}

@misc{ja_leaderboard_mgsm_2,
  title         = {Language Models are Multilingual Chain-of-Thought Reasoners},
  author        = {
    Freda Shi and Mirac Suzgun and Markus Freitag and Xuezhi Wang and Suraj Srivats and Soroush
    Vosoughi and Hyung Won Chung and Yi Tay and Sebastian Ruder and Denny Zhou and Dipanjan Das and
    Jason Wei
  },
  year          = 2022,
  eprint        = {2210.03057},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CL}
}

@inproceedings{ja_leaderboard_xlsum,
  title         = {{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages},
  author        = {
    Hasan, Tahmid  and Bhattacharjee, Abhik  and Islam, Md. Saiful  and Mubasshir, Kazi  and Li,
    Yuan-Fang  and Kang, Yong-Bin  and Rahman, M. Sohel  and Shahriyar, Rifat
  },
  year          = 2021,
  month         = aug,
  booktitle     = {Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  publisher     = {Association for Computational Linguistics},
  address       = {Online},
  pages         = {4693--4703},
  url           = {https://aclanthology.org/2021.findings-acl.413}
}

@article{jglue_2023,
  title         = {JGLUE: 日本語言語理解ベンチマーク},
  author        = {栗原 健太郎 and 河原 大輔 and 柴田 知秀},
  year          = 2023,
  journal       = {自然言語処理},
  volume        = 30,
  number        = 1,
  pages         = {63--87},
  doi           = {10.5715/jnlp.30.63},
  url           = {https://www.jstage.jst.go.jp/article/jnlp/30/1/30_63/_article/-char/ja}
}

@inproceedings{jglue_kurihara-etal-2022-jglue,
  title         = {{JGLUE}: {J}apanese General Language Understanding Evaluation},
  author        = {Kurihara, Kentaro  and Kawahara, Daisuke  and Shibata, Tomohide},
  year          = 2022,
  month         = jun,
  booktitle     = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  publisher     = {European Language Resources Association},
  address       = {Marseille, France},
  pages         = {2957--2966},
  url           = {https://aclanthology.org/2022.lrec-1.317}
}

@inproceedings{jglue_kurihara_nlp2022,
  title         = {JGLUE: 日本語言語理解ベンチマーク},
  author        = {栗原健太郎 and 河原大輔 and 柴田知秀},
  year          = 2022,
  booktitle     = {言語処理学会第28回年次大会},
  url           = {https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E8-4.pdf},
  note          = {in Japanese}
}

@misc{xwinograd_muennighoff2022crosslingual,
  title         = {Crosslingual Generalization through Multitask Finetuning},
  author        = {
    Niklas Muennighoff and Thomas Wang and Lintang Sutawika and Adam Roberts and Stella Biderman
    and Teven Le Scao and M Saiful Bari and Sheng Shen and Zheng-Xin Yong and Hailey Schoelkopf and
    Xiangru Tang and Dragomir Radev and Alham Fikri Aji and Khalid Almubarak and Samuel Albanie and
    Zaid Alyafeai and Albert Webson and Edward Raff and Colin Raffel
  },
  year          = 2022,
  eprint        = {2211.01786},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CL}
}

@misc{xwinograd_tikhonov2021heads,
  title         = {
    It's All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in
    Commonsense Reasoning
  },
  author        = {Alexey Tikhonov and Max Ryabinin},
  year          = 2021,
  eprint        = {2106.12066},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CL}
}
```

### Credit

* Prompts: https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable/lm_eval/tasks/ja

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
