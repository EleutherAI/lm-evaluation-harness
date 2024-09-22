# BlueBench Benchmark

BlueBench is an open-source benchmark developed by domain experts to represent required needs of Enterprise users. It is constructed using state-of-the-art benchmarking methodologies to ensure validity, robustness, and efficiency. As a dynamic and evolving benchmark, BlueBench currently encompasses diverse domains such as legal, finance, customer support, and news. It also evaluates a range of capabilities, including RAG, pro-social behavior, summarization, and chatbot performance, with additional tasks and domains to be integrated over time.

### Groups, Tags, and Tasks

#### Groups

The 13 BlueBench scenarios

* `bluebench_reasoning`
* `bluebench_translation`
* `bluebench_chatbot_abilities`
* `bluebench_news_classification`
* `bluebench_bias`
* `bluebench_legal`
* `bluebench_product_help`
* `bluebench_knowledge`
* `bluebench_entity_extraction`
* `bluebench_safety`
* `bluebench_summarization`
* `bluebench_rag_general`
* `bluebench_rag_finance`

#### Tags

None.

#### Tasks

The 57 BlueBench sub-scenarios
Naming convention: 'bluebench_{scenario}_{sub-scenario}'

* `bluebench_reasoning_hellaswag`
* `bluebench_reasoning_openbook_qa`
* `bluebench_translation_mt_flores_101_ara_eng`
* `bluebench_translation_mt_flores_101_deu_eng`
* `bluebench_translation_mt_flores_101_eng_ara`
* `bluebench_translation_mt_flores_101_eng_deu`
* `bluebench_translation_mt_flores_101_eng_fra`
* `bluebench_translation_mt_flores_101_eng_kor`
* `bluebench_translation_mt_flores_101_eng_por`
* `bluebench_translation_mt_flores_101_eng_ron`
* `bluebench_translation_mt_flores_101_eng_spa`
* `bluebench_translation_mt_flores_101_fra_eng`
* `bluebench_translation_mt_flores_101_jpn_eng`
* `bluebench_translation_mt_flores_101_kor_eng`
* `bluebench_translation_mt_flores_101_por_eng`
* `bluebench_translation_mt_flores_101_ron_eng`
* `bluebench_translation_mt_flores_101_spa_eng`
* `bluebench_chatbot_abilities_cards_arena_hard_generation_english_gpt_4_0314_reference`
* `bluebench_news_classification_20_newsgroups`
* `bluebench_bias_safety_bbq_age`
* `bluebench_bias_safety_bbq_disability_status`
* `bluebench_bias_safety_bbq_gender_identity`
* `bluebench_bias_safety_bbq_nationality`
* `bluebench_bias_safety_bbq_physical_appearance`
* `bluebench_bias_safety_bbq_race_ethnicity`
* `bluebench_bias_safety_bbq_race_x_ses`
* `bluebench_bias_safety_bbq_race_x_gender`
* `bluebench_bias_safety_bbq_religion`
* `bluebench_bias_safety_bbq_ses`
* `bluebench_bias_safety_bbq_sexual_orientation`
* `bluebench_legal_legalbench_abercrombie`
* `bluebench_legal_legalbench_proa`
* `bluebench_legal_legalbench_function_of_decision_section`
* `bluebench_legal_legalbench_international_citizenship_questions`
* `bluebench_legal_legalbench_corporate_lobbying`
* `bluebench_product_help_cfpb_product_watsonx`
* `bluebench_product_help_cfpb_product_2023`
* `bluebench_knowledge_mmlu_pro_history`
* `bluebench_knowledge_mmlu_pro_law`
* `bluebench_knowledge_mmlu_pro_health`
* `bluebench_knowledge_mmlu_pro_physics`
* `bluebench_knowledge_mmlu_pro_business`
* `bluebench_knowledge_mmlu_pro_other`
* `bluebench_knowledge_mmlu_pro_philosophy`
* `bluebench_knowledge_mmlu_pro_psychology`
* `bluebench_knowledge_mmlu_pro_economics`
* `bluebench_knowledge_mmlu_pro_math`
* `bluebench_knowledge_mmlu_pro_biology`
* `bluebench_knowledge_mmlu_pro_chemistry`
* `bluebench_knowledge_mmlu_pro_computer_science`
* `bluebench_knowledge_mmlu_pro_engineering`
* `bluebench_entity_extraction_cards_universal_ner_en_ewt`
* `bluebench_safety_attaq_500`
* `bluebench_summarization_billsum_document_filtered_to_6000_chars`
* `bluebench_summarization_tldr_document_filtered_to_6000_chars`
* `bluebench_rag_general_rag_response_generation_clapnq`
* `bluebench_rag_finance_fin_qa`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
