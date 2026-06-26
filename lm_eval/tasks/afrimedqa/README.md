# AfriMed-QA

### Paper

Title: AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset

Paper: https://arxiv.org/abs/2411.15640

Homepage: https://huggingface.co/datasets/intronhealth/afrimedqa_v2

AfriMed-QA is a Pan-African English medical question-answering benchmark with multiple-choice, short-answer, and consumer-query tasks collected across African medical contexts. The benchmark is designed to evaluate LLM medical knowledge, regional generalization, and response quality across specialties and geographies.

### Citation

```bibtex
@misc{olatunji2025afrimedqa,
  title={AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset},
  author={Olatunji, Tobi and Nimo, Charles and Owodunni, Abraham and Abdullahi, Tassallah and Ayodele, Emmanuel and Sanni, Mardhiyah and Aka, Chinemelu and Omofoye, Folafunmi and Yuehgoh, Foutse and Faniran, Timothy and Dossou, Bonaventure F. P. and Yekini, Moshood and Kemp, Jonas and Heller, Katherine and Omeke, Jude Chidubem and Asuzu, Chidi and Etori, Naome A. and Ndiaye, Aimerou and Okoh, Ifeoma and Ocansey, Evans Doe and Kinara, Wendy and Best, Michael and Essa, Irfan and Moore, Stephen Edward and Fourie, Chris and Asiedu, Mercy Nyamewaa},
  year={2025},
  eprint={2411.15640},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2411.15640}
}
```

### Groups and Tasks

* `afrimedqa`: Runs `afrimedqa_mcq`.

#### Tasks

* `afrimedqa_mcq`: Expert multiple-choice questions evaluated with generative letter extraction and accuracy.
