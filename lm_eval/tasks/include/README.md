# INCLUDE

### Paper

Title: `INCLUDE: Evaluating Multilingual Language Understanding with Regional Knowledge`

Abstract: [https://arxiv.org/abs/2411.19799](https://arxiv.org/abs/2411.19799)

INCLUDE is a comprehensive knowledge- and reasoning-centric benchmark across 44 languages that evaluates multilingual LLMs for performance in the actual language environments where they would be deployed. It contains 22,637 4-option multiple-choice-questions (MCQ) extracted from academic and professional exams, covering 57 topics, including regional knowledge.

> 🤗 [CohereForAI/include-base-44](https://huggingface.co/datasets/CohereForAI/include-base-44): Benchmark which supports 44 languages, each with 500 regional samples and 50 region-agnostic ones.


### Tasks:
We add the following evaluations:
- prompting with instructions in english for the 0-shot setting (`default`)
- prompting with instructions in english for the 5-shot setting (`few_shot_en`)
- prompting with instructions in the in-sample language for the 5-shot setting (`few_shot_og`)


### Languages

Albanian, Arabic, Armenian, Azerbaijani, Basque, Belarusian, Bengali, Bulgarian, Chinese, Croatian, Dutch, Estonian, Finnish, French, Georgian, German, Greek, Hebrew, Hindi, Hungarian, Indonesia, Italian, Japanese, Kazakh, Korean, Lithuanian, Malay, Malayalam, Nepali, North Macedonian, Persian, Polish, Portuguese, russian, Serbian, Spanish, Tagalog, Tamil, Telugu, Turkish, Ukrainian, Urdu, Uzbek, Vietnamese


### Domains

**Academic:** Accounting, Agriculture, Anthropology, Architecture and Design, Arts & Humanities, Biology, Business administration, Business ethics, Business, Chemistry, Computer Science, Culturology, Earth science, Economics, Education, Engineering, Environmental studies and forestry, Family and consumer science, Finance, Geography, Health, History, Human physical performance and recreation, Industrial and labor relations, International trade, Journalism, media studies, and communication, Language, Law, Library and museum studies, Literature, Logic, Management, Marketing, Math, Medicine, Military Sciences, Multiple exams, Performing arts, Philosophy, Physics, Political sciences, Psychology, Public Administration, Public Policy, Qualimetry, Religious studies, Risk management and insurance, Social Work, Social work, Sociology, STEM, Transportation, Visual Arts

**Licenses:** Driving License, Marine License, Medical License, Professional Certifications


### Changelog

- **v0.1**: Renamed `few_shot_en` and `few_shot_og` variant group names from `include_base_44_<lang>` to `include_base_44_<lang>_few_shot_en` and `include_base_44_<lang>_few_shot_og` respectively, to resolve duplicate group name conflicts with the `default` variants.

### Citation

```bibtex
 @article{romanou2024include,
  title={INCLUDE: Evaluating Multilingual Language Understanding with Regional Knowledge},
  author={Angelika Romanou and Negar Foroutan and Anna Sotnikova and Zeming Chen and Sree Harsha Nelaturu and Shivalika Singh and Rishabh Maheshwary and Micol Altomare and Mohamed A Haggag and Imanol Schlag and Marzieh Fadaee and Sara Hooker and Antoine Bosselut and others},
  journal={ICLR},
  year={2024},
  primaryClass={cs.CL},
  eprint={2411.19799},
  url={https://arxiv.org/abs/2411.19799},
}
```
