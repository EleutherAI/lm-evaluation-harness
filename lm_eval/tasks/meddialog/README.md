# Meddialog

### Paper

Title: `MedDialog: Large-scale Medical Dialogue Datasets`

Abstract: [https://aclanthology.org/2020.emnlp-main.743/](https://aclanthology.org/2020.emnlp-main.743/)

This task contains the english version of the MedDialog Medical Dialogue Dataset, divided in two tasks:
question entailment, and open-ended Question Answering (QA).

#### Tasks

* `meddialog_qsumm`: Question entailment in english.
* `meddialog_qsumm_perplexity`: Question entailment in english, evaluated with perplexity.
* `meddialog_raw_dialogues`: Open-Ended QA in english.
* `meddialog_raw_perplexity`: Open-Ended QA in english, evaluated with perplexity.

### Citation

```bibtex
@inproceedings{zeng-etal-2020-meddialog,
    title = "{M}ed{D}ialog: Large-scale Medical Dialogue Datasets",
    author = "Zeng, Guangtao  and
      Yang, Wenmian  and
      Ju, Zeqian  and
      Yang, Yue  and
      Wang, Sicheng  and
      Zhang, Ruisi  and
      Zhou, Meng  and
      Zeng, Jiaqi  and
      Dong, Xiangyu  and
      Zhang, Ruoyu  and
      Fang, Hongchao  and
      Zhu, Penghui  and
      Chen, Shu  and
      Xie, Pengtao",
    editor = "Webber, Bonnie  and
      Cohn, Trevor  and
      He, Yulan  and
      Liu, Yang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.743/",
    doi = "10.18653/v1/2020.emnlp-main.743",
    pages = "9241--9250",
    abstract = "Medical dialogue systems are promising in assisting in telemedicine to increase access to healthcare services, improve the quality of patient care, and reduce medical costs. To facilitate the research and development of medical dialogue systems, we build large-scale medical dialogue datasets {--} MedDialog, which contain 1) a Chinese dataset with 3.4 million conversations between patients and doctors, 11.3 million utterances, 660.2 million tokens, covering 172 specialties of diseases, and 2) an English dataset with 0.26 million conversations, 0.51 million utterances, 44.53 million tokens, covering 96 specialties of diseases. To our best knowledge, MedDialog is the largest medical dialogue dataset to date. We pretrain several dialogue generation models on the Chinese MedDialog dataset, including Transformer, GPT, BERT-GPT, and compare their performance. It is shown that models trained on MedDialog are able to generate clinically correct and doctor-like medical dialogues. We also study the transferability of models trained on MedDialog to low-resource medical dialogue generation tasks. It is shown that via transfer learning which finetunes the models pretrained on MedDialog, the performance on medical dialogue generation tasks with small datasets can be greatly improved, as shown in human evaluation and automatic evaluation. The datasets and code are available at \url{https://github.com/UCSD-AI4H/Medical-Dialogue-System}"
}
```
