# Multilingual TruthfulQA

### Paper

Title: `Okapi: Instruction-tuned Large Language Models in Multiple Languages with Reinforcement Learning from Human Feedback`

Abstract: https://arxiv.org/abs/2307.16039

A key technology for the development of large language models (LLMs) involves instruction tuning that helps align the models' responses with human expectations to realize impressive learning abilities. Two major approaches for instruction tuning characterize supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF), which are currently applied to produce the best commercial LLMs (e.g., ChatGPT). To improve the accessibility of LLMs for research and development efforts, various instruction-tuned open-source LLMs have also been introduced recently, e.g., Alpaca, Vicuna, to name a few. However, existing open-source LLMs have only been instruction-tuned for English and a few popular languages, thus hindering their impacts and accessibility to many other languages in the world. Among a few very recent work to explore instruction tuning for LLMs in multiple languages, SFT has been used as the only approach to instruction-tune LLMs for multiple languages. This has left a significant gap for fine-tuned LLMs based on RLHF in diverse languages and raised important questions on how RLHF can boost the performance of multilingual instruction tuning. To overcome this issue, we present Okapi, the first system with instruction-tuned LLMs based on RLHF for multiple languages. Okapi introduces instruction and response-ranked data in 26 diverse languages to facilitate the experiments and development of future multilingual LLM research. We also present benchmark datasets to enable the evaluation of generative LLMs in multiple languages. Our experiments demonstrate the advantages of RLHF for multilingual instruction over SFT for different base models and datasets. Our framework and resources are released at this https URL.

Homepage: `https://github.com/nlp-uoregon/Okapi`


### Citation

```
@article{dac2023okapi,
  title={Okapi: Instruction-tuned Large Language Models in Multiple Languages with Reinforcement Learning from Human Feedback},
  author={Dac Lai, Viet and Van Nguyen, Chien and Ngo, Nghia Trung and Nguyen, Thuat and Dernoncourt, Franck and Rossi, Ryan A and Nguyen, Thien Huu},
  journal={arXiv e-prints},
  pages={arXiv--2307},
  year={2023}
}
```

### Groups and Tasks

#### Groups

- truthfulqa_multilingual

#### Tasks

- `truthfulqa_{ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh}`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
