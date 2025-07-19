# SwitzerlandQA

Dataset: [swiss-ai/switzerland_qa](https://huggingface.co/datasets/swiss-ai/switzerland_qa)

SwitzerlandQA is a MCQ multilingual benchmark of exams collected from Switzerland.

### Dataset Overview

**Size:**
- 📝 9,167 questions per language
- 🌍 5 languages: English, German, French, Italian, Romansh
- 🏔️ At least 200 questions per canton (exact number varies based on resources)

**Question Format:**
- Multiple choice

**Collection:**
- 📚 Collected from naturalisation exams per canton; generated when unavailable from manually selected resources available on the official canton websites or provided by the culture departments.


#### Groups

* `switzerland_qa`: This group provides an aggregated score for the whole benchmark.
* `switzerland_qa_{lang}`: This group provides an aggregated score per language and supports 5 languages (en, fr, it, de, rm).

#### Subgroups per language

* `switzerland_qa_{lang}_geography`
* `switzerland_qa_{lang}_history`
* `switzerland_qa_{lang}_insurance`
* `switzerland_qa_{lang}_political_life`
* `switzerland_qa_{lang}_social_life`
