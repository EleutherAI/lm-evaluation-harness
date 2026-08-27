# Arab Culture

### Paper

Title: Commonsense Reasoning in Arab Culture


Abstract: https://arxiv.org/abs/2502.12788

Despite progress in Arabic large language models, such as Jais and AceGPT, their evaluation on commonsense reasoning has largely relied on machine-translated datasets, which lack cultural depth and may introduce Anglocentric biases. Commonsense reasoning is shaped by geographical and cultural contexts, and existing English datasets fail to capture the diversity of the Arab world. To address this, we introduce \datasetname, a commonsense reasoning dataset in Modern Standard Arabic (MSA), covering cultures of 13 countries across the Gulf, Levant, North Africa, and the Nile Valley. The dataset was built from scratch by engaging native speakers to write and validate culturally relevant questions for their respective countries. \datasetname spans 12 daily life domains with 54 fine-grained subtopics, reflecting various aspects of social norms, traditions, and everyday experiences. Zero-shot evaluations show that open-weight language models with up to 32B parameters struggle to comprehend diverse Arab cultures, with performance varying across regions. These findings highlight the need for more culturally aware models and datasets tailored to the Arabic-speaking world.

Homepage: https://github.com/fajri91/ArabicCulture


### Citation

```
@misc{sadallah2025commonsensereasoningarabculture,
      title={Commonsense Reasoning in Arab Culture},
      author={Abdelrahman Sadallah and Junior Cedric Tonga and Khalid Almubarak and Saeed Almheiri and Farah Atif and Chatrine Qwaider and Karima Kadaoui and Sara Shatnawi and Yaser Alesh and Fajri Koto},
      year={2025},
      eprint={2502.12788},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.12788},
}
```

### There are two variant of this task: `arab_culture`, and `arab_culture_completion`

- The `arab_culture` is the normal MCQ evaluation type, which appends the answers to the question, and then measure the likelihood of the different choices markers (A,B,C or "أ","ب","ج"). For more info, follow the MMLU style [tempelate](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_default_template_yaml#L7-L8)
- The `arab_culture_completion` do the evaluation in a sentence completion manner, by appending each asnwer to the question separetley and chooses the answer with the higher likelihood. See [this](https://github.com/EleutherAI/lm-evaluation-harness/blob/1f9bc88fe61f6bfa36f74e91ce3d59ab5685e4f1/lm_eval/tasks/arc/arc_easy.yaml#L10-L12) for more information

### Groups and Tasks

#### Groups

* `arabculture`: evaluates all ArabCulture tasks.

* `arab_culture_gulf`: evaluates Gulf countires ArabCulture tasks.
* `arab_culture_levant`: evaluates Levant countires ArabCulture tasks.
* `arab_culture_nile_valley`: evaluates Nile Valley countires ArabCulture tasks.
* `arab_culture_north_africa`: evaluates North Africa ArabCulture tasks.

###  Evaluation modes
This bechmark allows for different evaluation settings by allowing to adding more extra context for the model:

We have three settings:
* without any information
```
COUNTRY=False
REGION=False
```
* with  only region information
```
COUNTRY=False
REGION=True
```
* with region and country information
```
COUNTRY=True
REGION=True
```

**Please add these flags add environment variables.**


* We also allow for prompting in English, which we found to achieve higher results on most of the evaluated models (please refer to our paper).

* To change the language of the prompt, Define the `ARABIC` environment variable.
