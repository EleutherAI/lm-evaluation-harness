"""
Tasks from "Multilingual Large Language Models Evaluation Benchmark"

Source: https://github.com/nlp-uoregon/mlmm-evaluation

This repo contains benchmark datasets and evaluation scripts for Multilingual Large Language Models (LLMs). These datasets can be used to evaluate the models across 26 different languages and encompass three distinct tasks: ARC, HellaSwag, and MMLU. This is released as a part of our [Okapi framework](https://github.com/nlp-uoregon/Okapi) for multilingual instruction-tuned LLMs with reinforcement learning from human feedback.

- [**ARC**](https://allenai.org/data/arc): A dataset with 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage research in advanced question-answering.
- [**HellaSwag**](https://allenai.org/data/hellaswag): HellaSWAG is a dataset for studying grounded commonsense inference. It consists of 70k multiple choice questions about grounded situations: each question comes from one of two domains *activitynet* or *wikihow* with four answer choices about what might happen next in the scene. The correct answer is the (real) sentence for the next event; the three incorrect answers are adversarially generated and human verified, so as to fool machines but not humans.
- [**MMLU**](https://arxiv.org/pdf/2009.03300.pdf): This dataset contains multiple choice questions derived from diverse fields of knowledge. The test covers subjects in the humanities, social sciences, hard sciences, and other essential areas of learning for certain individuals.

Currently, our datasets support 26 languages: Russian, German, Chinese, French, Spanish, Italian, Dutch, Vietnamese, Indonesian, Arabic, Hungarian, Romanian, Danish, Slovak, Ukrainian, Catalan, Serbian, Croatian, Hindi, Bengali, Tamil, Nepali, Malayalam, Marathi, Telugu, and Kannada.

"""
