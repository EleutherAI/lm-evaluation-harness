# CMMLU

### Paper

CMMLU: Measuring massive multitask language understanding in Chinese
https://arxiv.org/abs/2306.09212

CMMLU is a comprehensive evaluation benchmark specifically designed to evaluate the knowledge and reasoning abilities of LLMs within the context of Chinese language and culture.
CMMLU covers a wide range of subjects, comprising 67 topics that span from elementary to advanced professional levels.

Homepage: https://github.com/haonan-li/CMMLU

### Citation

```bibtex
@misc{li2023cmmlu,
      title={CMMLU: Measuring massive multitask language understanding in Chinese},
      author={Haonan Li and Yixuan Zhang and Fajri Koto and Yifei Yang and Hai Zhao and Yeyun Gong and Nan Duan and Timothy Baldwin},
      year={2023},
      eprint={2306.09212},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


|                   Tasks                    |Version|Filter|       Metric       |Value |   |Stderr|
|--------------------------------------------|-------|------|--------------------|-----:|---|-----:|
|cmmlu                                       |N/A    |none  |acc                 |0.2480|   |      |
|                                            |       |none  |acc(sample agg)     |0.2494|   |      |
|                                            |       |none  |acc_norm            |0.2480|   |      |
|                                            |       |none  |acc_norm(sample agg)|0.2494|   |      |
|-cmmlu_modern_chinese                       |Yaml   |none  |acc                 |0.2500|±  |0.0404|
|                                            |       |none  |acc_norm            |0.2500|±  |0.0404|
|-cmmlu_world_history                        |Yaml   |none  |acc                 |0.2484|±  |0.0342|
|                                            |       |none  |acc_norm            |0.2484|±  |0.0342|
|-cmmlu_college_education                    |Yaml   |none  |acc                 |0.2523|±  |0.0422|
|                                            |       |none  |acc_norm            |0.2523|±  |0.0422|
|-cmmlu_international_law                    |Yaml   |none  |acc                 |0.2486|±  |0.0319|
|                                            |       |none  |acc_norm            |0.2486|±  |0.0319|
|-cmmlu_philosophy                           |Yaml   |none  |acc                 |0.1905|±  |0.0385|
|                                            |       |none  |acc_norm            |0.1905|±  |0.0385|
|-cmmlu_professional_psychology              |Yaml   |none  |acc                 |0.2457|±  |0.0283|
|                                            |       |none  |acc_norm            |0.2457|±  |0.0283|
|-cmmlu_college_engineering_hydrology        |Yaml   |none  |acc                 |0.2830|±  |0.0440|
|                                            |       |none  |acc_norm            |0.2830|±  |0.0440|
|-cmmlu_electrical_engineering               |Yaml   |none  |acc                 |0.2442|±  |0.0329|
|                                            |       |none  |acc_norm            |0.2442|±  |0.0329|
|-cmmlu_ancient_chinese                      |Yaml   |none  |acc                 |0.2378|±  |0.0333|
|                                            |       |none  |acc_norm            |0.2378|±  |0.0333|
|-cmmlu_chinese_food_culture                 |Yaml   |none  |acc                 |0.2353|±  |0.0365|
|                                            |       |none  |acc_norm            |0.2353|±  |0.0365|
|-cmmlu_chinese_literature                   |Yaml   |none  |acc                 |0.2598|±  |0.0308|
|                                            |       |none  |acc_norm            |0.2598|±  |0.0308|
|-cmmlu_legal_and_moral_basis                |Yaml   |none  |acc                 |0.2477|±  |0.0296|
|                                            |       |none  |acc_norm            |0.2477|±  |0.0296|
|-cmmlu_construction_project_management      |Yaml   |none  |acc                 |0.2374|±  |0.0362|
|                                            |       |none  |acc_norm            |0.2374|±  |0.0362|
|-cmmlu_ethnology                            |Yaml   |none  |acc                 |0.2519|±  |0.0375|
|                                            |       |none  |acc_norm            |0.2519|±  |0.0375|
|-cmmlu_high_school_geography                |Yaml   |none  |acc                 |0.2542|±  |0.0403|
|                                            |       |none  |acc_norm            |0.2542|±  |0.0403|
|-cmmlu_professional_medicine                |Yaml   |none  |acc                 |0.2500|±  |0.0224|
|                                            |       |none  |acc_norm            |0.2500|±  |0.0224|
|-cmmlu_global_facts                         |Yaml   |none  |acc                 |0.2349|±  |0.0348|
|                                            |       |none  |acc_norm            |0.2349|±  |0.0348|
|-cmmlu_astronomy                            |Yaml   |none  |acc                 |0.2303|±  |0.0329|
|                                            |       |none  |acc_norm            |0.2303|±  |0.0329|
|-cmmlu_machine_learning                     |Yaml   |none  |acc                 |0.2541|±  |0.0396|
|                                            |       |none  |acc_norm            |0.2541|±  |0.0396|
|-cmmlu_high_school_politics                 |Yaml   |none  |acc                 |0.2378|±  |0.0357|
|                                            |       |none  |acc_norm            |0.2378|±  |0.0357|
|-cmmlu_chinese_civil_service_exam           |Yaml   |none  |acc                 |0.2562|±  |0.0346|
|                                            |       |none  |acc_norm            |0.2562|±  |0.0346|
|-cmmlu_professional_law                     |Yaml   |none  |acc                 |0.2512|±  |0.0299|
|                                            |       |none  |acc_norm            |0.2512|±  |0.0299|
|-cmmlu_college_medical_statistics           |Yaml   |none  |acc                 |0.2453|±  |0.0420|
|                                            |       |none  |acc_norm            |0.2453|±  |0.0420|
|-cmmlu_computer_security                    |Yaml   |none  |acc                 |0.2573|±  |0.0335|
|                                            |       |none  |acc_norm            |0.2573|±  |0.0335|
|-cmmlu_food_science                         |Yaml   |none  |acc                 |0.2238|±  |0.0350|
|                                            |       |none  |acc_norm            |0.2238|±  |0.0350|
|-cmmlu_security_study                       |Yaml   |none  |acc                 |0.2519|±  |0.0375|
|                                            |       |none  |acc_norm            |0.2519|±  |0.0375|
|-cmmlu_high_school_physics                  |Yaml   |none  |acc                 |0.2545|±  |0.0417|
|                                            |       |none  |acc_norm            |0.2545|±  |0.0417|
|-cmmlu_management                           |Yaml   |none  |acc                 |0.2476|±  |0.0299|
|                                            |       |none  |acc_norm            |0.2476|±  |0.0299|
|-cmmlu_professional_accounting              |Yaml   |none  |acc                 |0.2514|±  |0.0329|
|                                            |       |none  |acc_norm            |0.2514|±  |0.0329|
|-cmmlu_human_sexuality                      |Yaml   |none  |acc                 |0.2222|±  |0.0372|
|                                            |       |none  |acc_norm            |0.2222|±  |0.0372|
|-cmmlu_marxist_theory                       |Yaml   |none  |acc                 |0.2487|±  |0.0315|
|                                            |       |none  |acc_norm            |0.2487|±  |0.0315|
|-cmmlu_agronomy                             |Yaml   |none  |acc                 |0.2426|±  |0.0331|
|                                            |       |none  |acc_norm            |0.2426|±  |0.0331|
|-cmmlu_chinese_teacher_qualification        |Yaml   |none  |acc                 |0.2626|±  |0.0330|
|                                            |       |none  |acc_norm            |0.2626|±  |0.0330|
|-cmmlu_genetics                             |Yaml   |none  |acc                 |0.2273|±  |0.0317|
|                                            |       |none  |acc_norm            |0.2273|±  |0.0317|
|-cmmlu_sports_science                       |Yaml   |none  |acc                 |0.2727|±  |0.0348|
|                                            |       |none  |acc_norm            |0.2727|±  |0.0348|
|-cmmlu_elementary_commonsense               |Yaml   |none  |acc                 |0.2424|±  |0.0305|
|                                            |       |none  |acc_norm            |0.2424|±  |0.0305|
|-cmmlu_logical                              |Yaml   |none  |acc                 |0.1951|±  |0.0359|
|                                            |       |none  |acc_norm            |0.1951|±  |0.0359|
|-cmmlu_chinese_history                      |Yaml   |none  |acc                 |0.2508|±  |0.0242|
|                                            |       |none  |acc_norm            |0.2508|±  |0.0242|
|-cmmlu_traditional_chinese_medicine         |Yaml   |none  |acc                 |0.2378|±  |0.0314|
|                                            |       |none  |acc_norm            |0.2378|±  |0.0314|
|-cmmlu_elementary_mathematics               |Yaml   |none  |acc                 |0.2609|±  |0.0290|
|                                            |       |none  |acc_norm            |0.2609|±  |0.0290|
|-cmmlu_nutrition                            |Yaml   |none  |acc                 |0.2552|±  |0.0363|
|                                            |       |none  |acc_norm            |0.2552|±  |0.0363|
|-cmmlu_chinese_foreign_policy               |Yaml   |none  |acc                 |0.1776|±  |0.0371|
|                                            |       |none  |acc_norm            |0.1776|±  |0.0371|
|-cmmlu_journalism                           |Yaml   |none  |acc                 |0.2616|±  |0.0336|
|                                            |       |none  |acc_norm            |0.2616|±  |0.0336|
|-cmmlu_jurisprudence                        |Yaml   |none  |acc                 |0.2506|±  |0.0214|
|                                            |       |none  |acc_norm            |0.2506|±  |0.0214|
|-cmmlu_sociology                            |Yaml   |none  |acc                 |0.2478|±  |0.0288|
|                                            |       |none  |acc_norm            |0.2478|±  |0.0288|
|-cmmlu_college_mathematics                  |Yaml   |none  |acc                 |0.2190|±  |0.0406|
|                                            |       |none  |acc_norm            |0.2190|±  |0.0406|
|-cmmlu_computer_science                     |Yaml   |none  |acc                 |0.2549|±  |0.0306|
|                                            |       |none  |acc_norm            |0.2549|±  |0.0306|
|-cmmlu_conceptual_physics                   |Yaml   |none  |acc                 |0.2517|±  |0.0359|
|                                            |       |none  |acc_norm            |0.2517|±  |0.0359|
|-cmmlu_elementary_chinese                   |Yaml   |none  |acc                 |0.2817|±  |0.0284|
|                                            |       |none  |acc_norm            |0.2817|±  |0.0284|
|-cmmlu_marketing                            |Yaml   |none  |acc                 |0.2500|±  |0.0324|
|                                            |       |none  |acc_norm            |0.2500|±  |0.0324|
|-cmmlu_high_school_chemistry                |Yaml   |none  |acc                 |0.2576|±  |0.0382|
|                                            |       |none  |acc_norm            |0.2576|±  |0.0382|
|-cmmlu_college_law                          |Yaml   |none  |acc                 |0.2315|±  |0.0408|
|                                            |       |none  |acc_norm            |0.2315|±  |0.0408|
|-cmmlu_chinese_driving_rule                 |Yaml   |none  |acc                 |0.2595|±  |0.0384|
|                                            |       |none  |acc_norm            |0.2595|±  |0.0384|
|-cmmlu_clinical_knowledge                   |Yaml   |none  |acc                 |0.2532|±  |0.0283|
|                                            |       |none  |acc_norm            |0.2532|±  |0.0283|
|-cmmlu_education                            |Yaml   |none  |acc                 |0.2761|±  |0.0351|
|                                            |       |none  |acc_norm            |0.2761|±  |0.0351|
|-cmmlu_high_school_mathematics              |Yaml   |none  |acc                 |0.2927|±  |0.0356|
|                                            |       |none  |acc_norm            |0.2927|±  |0.0356|
|-cmmlu_college_actuarial_science            |Yaml   |none  |acc                 |0.2736|±  |0.0435|
|                                            |       |none  |acc_norm            |0.2736|±  |0.0435|
|-cmmlu_arts                                 |Yaml   |none  |acc                 |0.2313|±  |0.0334|
|                                            |       |none  |acc_norm            |0.2313|±  |0.0334|
|-cmmlu_public_relations                     |Yaml   |none  |acc                 |0.2471|±  |0.0328|
|                                            |       |none  |acc_norm            |0.2471|±  |0.0328|
|-cmmlu_college_medicine                     |Yaml   |none  |acc                 |0.2418|±  |0.0260|
|                                            |       |none  |acc_norm            |0.2418|±  |0.0260|
|-cmmlu_economics                            |Yaml   |none  |acc                 |0.2453|±  |0.0342|
|                                            |       |none  |acc_norm            |0.2453|±  |0.0342|
|-cmmlu_elementary_information_and_technology|Yaml   |none  |acc                 |0.2731|±  |0.0289|
|                                            |       |none  |acc_norm            |0.2731|±  |0.0289|
|-cmmlu_anatomy                              |Yaml   |none  |acc                 |0.2432|±  |0.0354|
|                                            |       |none  |acc_norm            |0.2432|±  |0.0354|
|-cmmlu_world_religions                      |Yaml   |none  |acc                 |0.2875|±  |0.0359|
|                                            |       |none  |acc_norm            |0.2875|±  |0.0359|
|-cmmlu_virology                             |Yaml   |none  |acc                 |0.2485|±  |0.0333|
|                                            |       |none  |acc_norm            |0.2485|±  |0.0333|
|-cmmlu_high_school_biology                  |Yaml   |none  |acc                 |0.2485|±  |0.0333|
|                                            |       |none  |acc_norm            |0.2485|±  |0.0333|
|-cmmlu_business_ethics                      |Yaml   |none  |acc                 |0.2584|±  |0.0304|
|                                            |       |none  |acc_norm            |0.2584|±  |0.0304|

|Groups|Version|Filter|       Metric       |Value |   |Stderr|
|------|-------|------|--------------------|-----:|---|------|
|cmmlu |N/A    |none  |acc                 |0.2480|   |      |
|      |       |none  |acc(sample agg)     |0.2494|   |      |
|      |       |none  |acc_norm            |0.2480|   |      |
|      |       |none  |acc_norm(sample agg)|0.2494|   |      |
