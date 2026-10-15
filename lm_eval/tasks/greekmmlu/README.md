# GreekMMLU

### Paper

Title: GreekMMLU: A Native-Sourced Multitask Benchmark for Evaluating Language Models in Greek


Abstract:
Large Language Models (LLMs) are commonly trained on multilingual corpora that include Greek, yet reliable evaluation benchmarks for Greek-particularly those based on authentic, native-sourced content-remain limited. Existing datasets are often machine-translated from English, failing to capture Greek linguistic and cultural characteristics. We introduce GreekMMLU, a native-sourced benchmark for massive multitask language understanding in Greek, comprising 21,805 multiple-choice questions across 45 subject areas, organized under a newly defined subject taxonomy and annotated with educational difficulty levels spanning primary to professional examinations. All questions are sourced or authored in Greek from academic, professional, and governmental exams. We publicly release 16,857 samples and reserve 4,948 samples for a private leaderboard to enable robust and contamination-resistant evaluation. Evaluations of over 80 open- and closed-source LLMs reveal substantial performance gaps between frontier and open-weight models, as well as between Greek-adapted models and general multilingual ones. Finally, we provide a systematic analysis of factors influencing performance-including model scale, adaptation, and prompting-and derive insights for improving LLM capabilities in Greek.

* **Paper:** https://arxiv.org/abs/2602.05150
* **Github Homepage:** https://github.com/mersinkonomi/GreekMMLU
* **Dataset on Hugging Face:** https://huggingface.co/datasets/dascim/GreekMMLU
* **Private leaderboard:** https://huggingface.co/spaces/yangzhang33/GreekMMLU-Leaderboard

### Citation

```bibtex
@article{zhang2026greekmmlu,
  title={GreekMMLU: A Native-Sourced Multitask Benchmark for Evaluating Language Models in Greek},
  author={Zhang, Yang and Konomi, Mersin and Xypolopoulos, Christos and Divriotis, Konstantinos and Skianis, Konstantinos and Nikolentzos, Giannis and Stamou, Giorgos and Shang, Guokan and Vazirgiannis, Michalis},
  journal={arXiv preprint arXiv:2602.05150},
  year={2026}
}
```


### Groups and Tasks

#### Groups

* `gmmlu`: evaluates all GreekMMLU tasks.

* `gmmlu_stem`: evaluates STEM GreekMMLU tasks.
* `gmmlu_social_sciences`: evaluates social science GreekMMLU tasks.
* `gmmlu_humanities`: evaluates humanities GreekMMLU tasks.
* `gmmlu_other`: evaluates other GreekMMLU tasks.
