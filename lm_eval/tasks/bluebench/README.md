# BlueBench Benchmark

BlueBench is an open-source benchmark developed by domain experts to represent required needs of Enterprise users. It is constructed using state-of-the-art benchmarking methodologies to ensure validity, robustness, and efficiency. As a dynamic and evolving benchmark, BlueBench currently encompasses diverse domains such as legal, finance, customer support, and news. It also evaluates a range of capabilities, including RAG, pro-social behavior, summarization, and chatbot performance, with additional tasks and domains to be integrated over time.

### Groups, Tags, and Tasks

#### Groups

The 13 BlueBench scenarios

* `bluebench_Reasoning`
* `bluebench_Translation`
* `bluebench_Chatbot_abilities`
* `bluebench_News_classification`
* `bluebench_Bias`
* `bluebench_Legal`
* `bluebench_Product_help`
* `bluebench_Knowledge`
* `bluebench_Entity_extraction`
* `bluebench_Safety`
* `bluebench_Summarization`
* `bluebench_RAG_general`
* `bluebench_RAG_finance`

#### Tags

None.

#### Tasks

The 57 BlueBench sub-scenarios
Naming convention: 'bluebench_{scenario}_{sub-scenario}'

* `bluebench_Reasoning_hellaswag`
* `bluebench_Reasoning_openbook_qa`
* `bluebench_Translation_mt_flores_101_ara_eng`
* `bluebench_Translation_mt_flores_101_deu_eng`
* `bluebench_Translation_mt_flores_101_eng_ara`
* `bluebench_Translation_mt_flores_101_eng_deu`
* `bluebench_Translation_mt_flores_101_eng_fra`
* `bluebench_Translation_mt_flores_101_eng_kor`
* `bluebench_Translation_mt_flores_101_eng_por`
* `bluebench_Translation_mt_flores_101_eng_ron`
* `bluebench_Translation_mt_flores_101_eng_spa`
* `bluebench_Translation_mt_flores_101_fra_eng`
* `bluebench_Translation_mt_flores_101_jpn_eng`
* `bluebench_Translation_mt_flores_101_kor_eng`
* `bluebench_Translation_mt_flores_101_por_eng`
* `bluebench_Translation_mt_flores_101_ron_eng`
* `bluebench_Translation_mt_flores_101_spa_eng`
* `bluebench_Chatbot_abilities_cards_arena_hard_generation_english_gpt_4_0314_reference`
* `bluebench_News_classification_20_newsgroups`
* `bluebench_Bias_safety_bbq_Age`
* `bluebench_Bias_safety_bbq_Disability_status`
* `bluebench_Bias_safety_bbq_Gender_identity`
* `bluebench_Bias_safety_bbq_Nationality`
* `bluebench_Bias_safety_bbq_Physical_appearance`
* `bluebench_Bias_safety_bbq_Race_ethnicity`
* `bluebench_Bias_safety_bbq_Race_x_SES`
* `bluebench_Bias_safety_bbq_Race_x_gender`
* `bluebench_Bias_safety_bbq_Religion`
* `bluebench_Bias_safety_bbq_SES`
* `bluebench_Bias_safety_bbq_Sexual_orientation`
* `bluebench_Legal_legalbench_abercrombie`
* `bluebench_Legal_legalbench_proa`
* `bluebench_Legal_legalbench_function_of_decision_section`
* `bluebench_Legal_legalbench_international_citizenship_questions`
* `bluebench_Legal_legalbench_corporate_lobbying`
* `bluebench_Product_help_CFPB_product_watsonx`
* `bluebench_Product_help_CFPB_product_2023`
* `bluebench_Knowledge_mmlu_pro_history`
* `bluebench_Knowledge_mmlu_pro_law`
* `bluebench_Knowledge_mmlu_pro_health`
* `bluebench_Knowledge_mmlu_pro_physics`
* `bluebench_Knowledge_mmlu_pro_business`
* `bluebench_Knowledge_mmlu_pro_other`
* `bluebench_Knowledge_mmlu_pro_philosophy`
* `bluebench_Knowledge_mmlu_pro_psychology`
* `bluebench_Knowledge_mmlu_pro_economics`
* `bluebench_Knowledge_mmlu_pro_math`
* `bluebench_Knowledge_mmlu_pro_biology`
* `bluebench_Knowledge_mmlu_pro_chemistry`
* `bluebench_Knowledge_mmlu_pro_computer_science`
* `bluebench_Knowledge_mmlu_pro_engineering`
* `bluebench_Entity_extraction_cards_universal_ner_en_ewt`
* `bluebench_Safety_attaq_500`
* `bluebench_Summarization_billsum_document_filtered_to_6000_chars`
* `bluebench_Summarization_tldr_document_filtered_to_6000_chars`
* `bluebench_RAG_general_rag_response_generation_clapnq`
* `bluebench_RAG_finance_fin_qa`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
