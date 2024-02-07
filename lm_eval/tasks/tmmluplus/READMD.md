---
license: other
license_name: creative-commons-by-nc
task_categories:
- question-answering
language:
- zh
tags:
- traditional chinese
- finance
- medical
- taiwan
- benchmark
- zh-tw
- zh-hant
pretty_name: tmmlu++
size_categories:
- 100K<n<1M
configs:
  - config_name: engineering_math
    datafiles:
    - split: train
      path: "data/engineering_math_dev.csv"
    - split: validation
      path: "data/engineering_math_val.csv"
    - split: test
      path: "data/engineering_math_test.csv"
  - config_name: dentistry
    datafiles:
    - split: train
      path: "data/dentistry_dev.csv"
    - split: validation
      path: "data/dentistry_val.csv"
    - split: test
      path: "data/dentistry_test.csv"
  - config_name: traditional_chinese_medicine_clinical_medicine
    datafiles:
    - split: train
      path: "data/traditional_chinese_medicine_clinical_medicine_dev.csv"
    - split: validation
      path: "data/traditional_chinese_medicine_clinical_medicine_val.csv"
    - split: test
      path: "data/traditional_chinese_medicine_clinical_medicine_test.csv"
  - config_name: clinical_psychology
    datafiles:
    - split: train
      path: "data/clinical_psychology_dev.csv"
    - split: validation
      path: "data/clinical_psychology_val.csv"
    - split: test
      path: "data/clinical_psychology_test.csv"
  - config_name: technical
    datafiles:
    - split: train
      path: "data/technical_dev.csv"
    - split: validation
      path: "data/technical_val.csv"
    - split: test
      path: "data/technical_test.csv"
  - config_name: culinary_skills
    datafiles:
    - split: train
      path: "data/culinary_skills_dev.csv"
    - split: validation
      path: "data/culinary_skills_val.csv"
    - split: test
      path: "data/culinary_skills_test.csv"
  - config_name: mechanical
    datafiles:
    - split: train
      path: "data/mechanical_dev.csv"
    - split: validation
      path: "data/mechanical_val.csv"
    - split: test
      path: "data/mechanical_test.csv"
  - config_name: logic_reasoning
    datafiles:
    - split: train
      path: "data/logic_reasoning_dev.csv"
    - split: validation
      path: "data/logic_reasoning_val.csv"
    - split: test
      path: "data/logic_reasoning_test.csv"
  - config_name: real_estate
    datafiles:
    - split: train
      path: "data/real_estate_dev.csv"
    - split: validation
      path: "data/real_estate_val.csv"
    - split: test
      path: "data/real_estate_test.csv"
  - config_name: general_principles_of_law
    datafiles:
    - split: train
      path: "data/general_principles_of_law_dev.csv"
    - split: validation
      path: "data/general_principles_of_law_val.csv"
    - split: test
      path: "data/general_principles_of_law_test.csv"
  - config_name: finance_banking
    datafiles:
    - split: train
      path: "data/finance_banking_dev.csv"
    - split: validation
      path: "data/finance_banking_val.csv"
    - split: test
      path: "data/finance_banking_test.csv"
  - config_name: anti_money_laundering
    datafiles:
    - split: train
      path: "data/anti_money_laundering_dev.csv"
    - split: validation
      path: "data/anti_money_laundering_val.csv"
    - split: test
      path: "data/anti_money_laundering_test.csv"
  - config_name: ttqav2
    datafiles:
    - split: train
      path: "data/ttqav2_dev.csv"
    - split: validation
      path: "data/ttqav2_val.csv"
    - split: test
      path: "data/ttqav2_test.csv"
  - config_name: marketing_management
    datafiles:
    - split: train
      path: "data/marketing_management_dev.csv"
    - split: validation
      path: "data/marketing_management_val.csv"
    - split: test
      path: "data/marketing_management_test.csv"
  - config_name: business_management
    datafiles:
    - split: train
      path: "data/business_management_dev.csv"
    - split: validation
      path: "data/business_management_val.csv"
    - split: test
      path: "data/business_management_test.csv"
  - config_name: organic_chemistry
    datafiles:
    - split: train
      path: "data/organic_chemistry_dev.csv"
    - split: validation
      path: "data/organic_chemistry_val.csv"
    - split: test
      path: "data/organic_chemistry_test.csv"
  - config_name: advance_chemistry
    datafiles:
    - split: train
      path: "data/advance_chemistry_dev.csv"
    - split: validation
      path: "data/advance_chemistry_val.csv"
    - split: test
      path: "data/advance_chemistry_test.csv"
  - config_name: physics
    datafiles:
    - split: train
      path: "data/physics_dev.csv"
    - split: validation
      path: "data/physics_val.csv"
    - split: test
      path: "data/physics_test.csv"
  - config_name: secondary_physics
    datafiles:
    - split: train
      path: "data/secondary_physics_dev.csv"
    - split: validation
      path: "data/secondary_physics_val.csv"
    - split: test
      path: "data/secondary_physics_test.csv"
  - config_name: human_behavior
    datafiles:
    - split: train
      path: "data/human_behavior_dev.csv"
    - split: validation
      path: "data/human_behavior_val.csv"
    - split: test
      path: "data/human_behavior_test.csv"
  - config_name: national_protection
    datafiles:
    - split: train
      path: "data/national_protection_dev.csv"
    - split: validation
      path: "data/national_protection_val.csv"
    - split: test
      path: "data/national_protection_test.csv"
  - config_name: jce_humanities
    datafiles:
    - split: train
      path: "data/jce_humanities_dev.csv"
    - split: validation
      path: "data/jce_humanities_val.csv"
    - split: test
      path: "data/jce_humanities_test.csv"
  - config_name: politic_science
    datafiles:
    - split: train
      path: "data/politic_science_dev.csv"
    - split: validation
      path: "data/politic_science_val.csv"
    - split: test
      path: "data/politic_science_test.csv"
  - config_name: agriculture
    datafiles:
    - split: train
      path: "data/agriculture_dev.csv"
    - split: validation
      path: "data/agriculture_val.csv"
    - split: test
      path: "data/agriculture_test.csv"
  - config_name: official_document_management
    datafiles:
    - split: train
      path: "data/official_document_management_dev.csv"
    - split: validation
      path: "data/official_document_management_val.csv"
    - split: test
      path: "data/official_document_management_test.csv"
  - config_name: financial_analysis
    datafiles:
    - split: train
      path: "data/financial_analysis_dev.csv"
    - split: validation
      path: "data/financial_analysis_val.csv"
    - split: test
      path: "data/financial_analysis_test.csv"
  - config_name: pharmacy
    datafiles:
    - split: train
      path: "data/pharmacy_dev.csv"
    - split: validation
      path: "data/pharmacy_val.csv"
    - split: test
      path: "data/pharmacy_test.csv"
  - config_name: educational_psychology
    datafiles:
    - split: train
      path: "data/educational_psychology_dev.csv"
    - split: validation
      path: "data/educational_psychology_val.csv"
    - split: test
      path: "data/educational_psychology_test.csv"
  - config_name: statistics_and_machine_learning
    datafiles:
    - split: train
      path: "data/statistics_and_machine_learning_dev.csv"
    - split: validation
      path: "data/statistics_and_machine_learning_val.csv"
    - split: test
      path: "data/statistics_and_machine_learning_test.csv"
  - config_name: management_accounting
    datafiles:
    - split: train
      path: "data/management_accounting_dev.csv"
    - split: validation
      path: "data/management_accounting_val.csv"
    - split: test
      path: "data/management_accounting_test.csv"
  - config_name: introduction_to_law
    datafiles:
    - split: train
      path: "data/introduction_to_law_dev.csv"
    - split: validation
      path: "data/introduction_to_law_val.csv"
    - split: test
      path: "data/introduction_to_law_test.csv"
  - config_name: computer_science
    datafiles:
    - split: train
      path: "data/computer_science_dev.csv"
    - split: validation
      path: "data/computer_science_val.csv"
    - split: test
      path: "data/computer_science_test.csv"
  - config_name: veterinary_pathology
    datafiles:
    - split: train
      path: "data/veterinary_pathology_dev.csv"
    - split: validation
      path: "data/veterinary_pathology_val.csv"
    - split: test
      path: "data/veterinary_pathology_test.csv"
  - config_name: accounting
    datafiles:
    - split: train
      path: "data/accounting_dev.csv"
    - split: validation
      path: "data/accounting_val.csv"
    - split: test
      path: "data/accounting_test.csv"
  - config_name: fire_science
    datafiles:
    - split: train
      path: "data/fire_science_dev.csv"
    - split: validation
      path: "data/fire_science_val.csv"
    - split: test
      path: "data/fire_science_test.csv"
  - config_name: optometry
    datafiles:
    - split: train
      path: "data/optometry_dev.csv"
    - split: validation
      path: "data/optometry_val.csv"
    - split: test
      path: "data/optometry_test.csv"
  - config_name: insurance_studies
    datafiles:
    - split: train
      path: "data/insurance_studies_dev.csv"
    - split: validation
      path: "data/insurance_studies_val.csv"
    - split: test
      path: "data/insurance_studies_test.csv"
  - config_name: pharmacology
    datafiles:
    - split: train
      path: "data/pharmacology_dev.csv"
    - split: validation
      path: "data/pharmacology_val.csv"
    - split: test
      path: "data/pharmacology_test.csv"
  - config_name: taxation
    datafiles:
    - split: train
      path: "data/taxation_dev.csv"
    - split: validation
      path: "data/taxation_val.csv"
    - split: test
      path: "data/taxation_test.csv"
  - config_name: trust_practice
    datafiles:
    - split: train
      path: "data/trust_practice_dev.csv"
    - split: validation
      path: "data/trust_practice_val.csv"
    - split: test
      path: "data/trust_practice_test.csv"
  - config_name: geography_of_taiwan
    datafiles:
    - split: train
      path: "data/geography_of_taiwan_dev.csv"
    - split: validation
      path: "data/geography_of_taiwan_val.csv"
    - split: test
      path: "data/geography_of_taiwan_test.csv"
  - config_name: physical_education
    datafiles:
    - split: train
      path: "data/physical_education_dev.csv"
    - split: validation
      path: "data/physical_education_val.csv"
    - split: test
      path: "data/physical_education_test.csv"
  - config_name: auditing
    datafiles:
    - split: train
      path: "data/auditing_dev.csv"
    - split: validation
      path: "data/auditing_val.csv"
    - split: test
      path: "data/auditing_test.csv"
  - config_name: administrative_law
    datafiles:
    - split: train
      path: "data/administrative_law_dev.csv"
    - split: validation
      path: "data/administrative_law_val.csv"
    - split: test
      path: "data/administrative_law_test.csv"
  - config_name: education_(profession_level)
    datafiles:
    - split: train
      path: "data/education_(profession_level)_dev.csv"
    - split: validation
      path: "data/education_(profession_level)_val.csv"
    - split: test
      path: "data/education_(profession_level)_test.csv"
  - config_name: economics
    datafiles:
    - split: train
      path: "data/economics_dev.csv"
    - split: validation
      path: "data/economics_val.csv"
    - split: test
      path: "data/economics_test.csv"
  - config_name: veterinary_pharmacology
    datafiles:
    - split: train
      path: "data/veterinary_pharmacology_dev.csv"
    - split: validation
      path: "data/veterinary_pharmacology_val.csv"
    - split: test
      path: "data/veterinary_pharmacology_test.csv"
  - config_name: nautical_science
    datafiles:
    - split: train
      path: "data/nautical_science_dev.csv"
    - split: validation
      path: "data/nautical_science_val.csv"
    - split: test
      path: "data/nautical_science_test.csv"
  - config_name: occupational_therapy_for_psychological_disorders
    datafiles:
    - split: train
      path: "data/occupational_therapy_for_psychological_disorders_dev.csv"
    - split: validation
      path: "data/occupational_therapy_for_psychological_disorders_val.csv"
    - split: test
      path: "data/occupational_therapy_for_psychological_disorders_test.csv"
  - config_name: basic_medical_science
    datafiles:
    - split: train
      path: "data/basic_medical_science_dev.csv"
    - split: validation
      path: "data/basic_medical_science_val.csv"
    - split: test
      path: "data/basic_medical_science_test.csv"
  - config_name: macroeconomics
    datafiles:
    - split: train
      path: "data/macroeconomics_dev.csv"
    - split: validation
      path: "data/macroeconomics_val.csv"
    - split: test
      path: "data/macroeconomics_test.csv"
  - config_name: trade
    datafiles:
    - split: train
      path: "data/trade_dev.csv"
    - split: validation
      path: "data/trade_val.csv"
    - split: test
      path: "data/trade_test.csv"
  - config_name: chinese_language_and_literature
    datafiles:
    - split: train
      path: "data/chinese_language_and_literature_dev.csv"
    - split: validation
      path: "data/chinese_language_and_literature_val.csv"
    - split: test
      path: "data/chinese_language_and_literature_test.csv"
  - config_name: tve_design
    datafiles:
    - split: train
      path: "data/tve_design_dev.csv"
    - split: validation
      path: "data/tve_design_val.csv"
    - split: test
      path: "data/tve_design_test.csv"
  - config_name: junior_science_exam
    datafiles:
    - split: train
      path: "data/junior_science_exam_dev.csv"
    - split: validation
      path: "data/junior_science_exam_val.csv"
    - split: test
      path: "data/junior_science_exam_test.csv"
  - config_name: junior_math_exam
    datafiles:
    - split: train
      path: "data/junior_math_exam_dev.csv"
    - split: validation
      path: "data/junior_math_exam_val.csv"
    - split: test
      path: "data/junior_math_exam_test.csv"
  - config_name: junior_chinese_exam
    datafiles:
    - split: train
      path: "data/junior_chinese_exam_dev.csv"
    - split: validation
      path: "data/junior_chinese_exam_val.csv"
    - split: test
      path: "data/junior_chinese_exam_test.csv"
  - config_name: junior_social_studies
    datafiles:
    - split: train
      path: "data/junior_social_studies_dev.csv"
    - split: validation
      path: "data/junior_social_studies_val.csv"
    - split: test
      path: "data/junior_social_studies_test.csv"
  - config_name: tve_mathematics
    datafiles:
    - split: train
      path: "data/tve_mathematics_dev.csv"
    - split: validation
      path: "data/tve_mathematics_val.csv"
    - split: test
      path: "data/tve_mathematics_test.csv"
  - config_name: tve_chinese_language
    datafiles:
    - split: train
      path: "data/tve_chinese_language_dev.csv"
    - split: validation
      path: "data/tve_chinese_language_val.csv"
    - split: test
      path: "data/tve_chinese_language_test.csv"
  - config_name: tve_natural_sciences
    datafiles:
    - split: train
      path: "data/tve_natural_sciences_dev.csv"
    - split: validation
      path: "data/tve_natural_sciences_val.csv"
    - split: test
      path: "data/tve_natural_sciences_test.csv"
  - config_name: junior_chemistry
    datafiles:
    - split: train
      path: "data/junior_chemistry_dev.csv"
    - split: validation
      path: "data/junior_chemistry_val.csv"
    - split: test
      path: "data/junior_chemistry_test.csv"
  - config_name: music
    datafiles:
    - split: train
      path: "data/music_dev.csv"
    - split: validation
      path: "data/music_val.csv"
    - split: test
      path: "data/music_test.csv"
  - config_name: education
    datafiles:
    - split: train
      path: "data/education_dev.csv"
    - split: validation
      path: "data/education_val.csv"
    - split: test
      path: "data/education_test.csv"
  - config_name: three_principles_of_people
    datafiles:
    - split: train
      path: "data/three_principles_of_people_dev.csv"
    - split: validation
      path: "data/three_principles_of_people_val.csv"
    - split: test
      path: "data/three_principles_of_people_test.csv"
  - config_name: taiwanese_hokkien
    datafiles:
    - split: train
      path: "data/taiwanese_hokkien_dev.csv"
    - split: validation
      path: "data/taiwanese_hokkien_val.csv"
    - split: test
      path: "data/taiwanese_hokkien_test.csv"
  - config_name: linear_algebra
    datafiles:
    - split: train
      path: "data/linear_algebra_dev.csv"
    - split: validation
      path: "data/linear_algebra_val.csv"
    - split: test
      path: "data/linear_algebra_test.csv"
---
# TMMLU+ : Large scale traditional chinese massive multitask language understanding

<p align="center">
<img src="https://huggingface.co/datasets/ikala/tmmluplus/resolve/main/cover.png" alt="A close-up image of a neat paper note with a white background. The text 'TMMLU+' is written horizontally across the center of the note in bold, black. Join us to work in multimodal LLM : https://ikala.ai/recruit/" style="max-width: 400" width=400 />
</p>
We present TMMLU+, a traditional Chinese massive multitask language understanding dataset. TMMLU+ is a multiple-choice question-answering dataset featuring 66 subjects, ranging from elementary to professional level.

The TMMLU+ dataset is six times larger and contains more balanced subjects compared to its predecessor, [TMMLU](https://github.com/mtkresearch/MR-Models/tree/main/TC-Eval/data/TMMLU). We have included benchmark results in TMMLU+ from closed-source models and 20 open-weight Chinese large language models, with parameters ranging from 1.8B to 72B. The benchmark results show that Traditional Chinese variants still lag behind those trained on major Simplified Chinese models.


```python
from datasets import load_dataset
task_list = [
             'engineering_math', 'dentistry', 'traditional_chinese_medicine_clinical_medicine', 'clinical_psychology', 'technical', 'culinary_skills', 'mechanical', 'logic_reasoning', 'real_estate',
             'general_principles_of_law', 'finance_banking', 'anti_money_laundering', 'ttqav2', 'marketing_management', 'business_management', 'organic_chemistry', 'advance_chemistry',
             'physics', 'secondary_physics', 'human_behavior', 'national_protection', 'jce_humanities', 'politic_science', 'agriculture', 'official_document_management',
             'financial_analysis', 'pharmacy', 'educational_psychology', 'statistics_and_machine_learning', 'management_accounting', 'introduction_to_law', 'computer_science', 'veterinary_pathology',
             'accounting', 'fire_science', 'optometry', 'insurance_studies', 'pharmacology', 'taxation', 'trust_practice', 'geography_of_taiwan', 'physical_education', 'auditing', 'administrative_law',
             'education_(profession_level)', 'economics', 'veterinary_pharmacology', 'nautical_science', 'occupational_therapy_for_psychological_disorders',
             'basic_medical_science', 'macroeconomics', 'trade', 'chinese_language_and_literature', 'tve_design', 'junior_science_exam', 'junior_math_exam', 'junior_chinese_exam',
             'junior_social_studies', 'tve_mathematics', 'tve_chinese_language', 'tve_natural_sciences', 'junior_chemistry', 'music', 'education', 'three_principles_of_people',
             'taiwanese_hokkien',
             'linear_algebra'
            ]
for task in task_list:
  val = load_dataset('ZoneTwelve/tmmluplus', task)['validation']
  dev = load_dataset('ZoneTwelve/tmmluplus', task)['train']
  test = load_dataset('ZoneTwelve/tmmluplus', task)['test']
```

For each dataset split

```python
for row in test:
  print(row)
  break
>> Dataset({
    features: ['question', 'A', 'B', 'C', 'D', 'answer'],
    num_rows: 11
})
```

Statistic on all four categories : STEM, Social Science, Humanities, Other

| Category                         | Test  | Dev  | Validation |
|----------------------------------|-------|------|------------|
| STEM                             | 3458  | 70   | 385        |
| Social Sciences                  | 5958  | 90   | 665        |
| Humanities                       | 1763  | 35   | 197        |
| Other (Business, Health, Misc.)  | 8939  | 135  | 995        |
| **Total**                        | 20118 | 330  | 2242       |


## Benchmark on direct prompting

| model | STEM | Social Science | Humanities | Other | Average |
|------------|------------|------------|------------|------------|------------|
| [Qwen/Qwen-72B](https://huggingface.co/Qwen/Qwen-72B) | 61.12 | 71.65 | 63.00 | 61.31 |64.27|
| gpt-4-0613 | 60.36 | 67.36 | 56.03 | 57.62 |60.34|
| [Qwen/Qwen-72B-Chat](https://huggingface.co/Qwen/Qwen-72B-Chat) | 55.15 | 66.20 | 55.65 | 57.19 |58.55|
| [Qwen/Qwen-14B](https://huggingface.co/Qwen/Qwen-14B) | 46.94 | 56.69 | 49.43 | 48.81 |50.47|
| Gemini-pro | 45.38 | 57.29 | 48.80 | 48.21 |49.92|
| [01-ai/Yi-34B-Chat](https://huggingface.co/01-ai/Yi-34B-Chat) | 40.24 | 56.77 | 53.99 | 47.58 |49.64|
| [Qwen/Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat) | 43.86 | 53.29 | 44.78 | 45.13 |46.77|
| [01-ai/Yi-6B-Chat](https://huggingface.co/01-ai/Yi-6B-Chat) | 39.62 | 50.24 | 44.44 | 44.26 |44.64|
| Claude-1.3 | 42.65 | 49.33 | 42.16 | 44.14 |44.57|
| gpt-3.5-turbo-0613 | 41.56 | 46.72 | 36.73 | 42.03 |41.76|
| [CausalLM/14B](https://huggingface.co/CausalLM/14B) | 39.83 | 44.50 | 39.61 | 41.97 |41.48|
| [Skywork/Skywork-13B-base](https://huggingface.co/Skywork/Skywork-13B-base) | 36.93 | 47.27 | 41.04 | 40.10 |41.33|
| [Qwen/Qwen-7B](https://huggingface.co/Qwen/Qwen-7B) | 37.53 | 45.48 | 38.09 | 38.96 |40.01|
| [Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat) | 33.32 | 44.64 | 40.27 | 39.89 |39.53|
| [vivo-ai/BlueLM-7B-Base](https://huggingface.co/vivo-ai/BlueLM-7B-Base) | 33.94 | 41.52 | 37.38 | 38.74 |37.90|
| [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) | 29.64 | 43.73 | 37.36 | 39.88 |37.65|
| [Qwen/Qwen-1_8B](https://huggingface.co/Qwen/Qwen-1_8B) | 32.65 | 38.95 | 38.34 | 35.27 |36.30|
| Claude-2 | 39.65 | 39.09 | 28.59 | 37.47 |36.20|
| [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) | 31.05 | 39.31 | 35.64 | 35.60 |35.40|
| [deepseek-ai/deepseek-llm-7b-chat](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) | 29.82 | 42.29 | 34.24 | 34.31 |35.17|
| [CausalLM/7B](https://huggingface.co/CausalLM/7B) | 31.03 | 38.17 | 35.87 | 35.39 |35.11|
| [Azure99/blossom-v3_1-mistral-7b](https://huggingface.co/Azure99/blossom-v3_1-mistral-7b) | 32.80 | 36.91 | 32.36 | 34.53 |34.15|
| [microsoft/Orca-2-13b](https://huggingface.co/microsoft/Orca-2-13b) | 24.69 | 39.18 | 33.60 | 31.99 |32.37|
| [Qwen/Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat) | 26.60 | 36.36 | 31.81 | 31.96 |31.68|
| [TigerResearch/tigerbot-13b-chat-v3](https://huggingface.co/TigerResearch/tigerbot-13b-chat-v3) | 24.73 | 29.63 | 25.72 | 27.22 |26.82|
| [hongyin/mistral-7b-80k](https://huggingface.co/hongyin/mistral-7b-80k) | 24.26 | 23.76 | 22.56 | 24.57 |23.79|
| [deepseek-ai/deepseek-llm-67b-chat](https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat) | 19.10 | 26.06 | 21.51 | 21.77 |22.11|
| [yentinglin/Taiwan-LLM-13B-v2.0-chat](https://huggingface.co/yentinglin/Taiwan-LLM-13B-v2.0-chat) | 18.53 | 27.65 | 17.77 | 21.49 |21.36|
| [GeneZC/MiniChat-3B](https://huggingface.co/GeneZC/MiniChat-3B) | 17.66 | 23.35 | 22.71 | 20.34 |21.02|
| [LinkSoul/Chinese-Llama-2-7b](https://huggingface.co/LinkSoul/Chinese-Llama-2-7b) | 16.55 | 18.39 | 12.97 | 16.13 |16.01|
| [yentinglin/Taiwan-LLM-7B-v2.1-chat](https://huggingface.co/yentinglin/Taiwan-LLM-7B-v2.1-chat) | 14.99 | 16.23 | 15.00 | 16.22 |15.61|
| Claude-instant-1 | 12.52 | 17.13 | 15.10 | 13.57 |14.58|
| [FlagAlpha/Atom-7B](https://huggingface.co/FlagAlpha/Atom-7B) | 5.60 | 13.57 | 7.71 | 11.84 |9.68|

Results via [ievals](https://github.com/iKala/ievals) ( settings : 0-shot direct answering )


# Citation

```
@article{ikala2023eval,
  title={An Improved Traditional Chinese Evaluation Suite for Foundation Model},
  author={Tam, Zhi-Rui and Pai, Ya-Ting},
  journal={arXiv},
  year={2023}
}
```

> CONTENT WARNING
> This is a modification of ikala/tmmluplus, with minor alterations made to facilitate the implementation for lm-evaluation-harness purposes.
> [More details on Discussions](https://huggingface.co/datasets/ZoneTwelve/tmmluplus/discussions/1)
