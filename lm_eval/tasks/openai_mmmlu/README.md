# OpenAI MMMLU

### Technical Report

The task/dataset contains a professional, human-translation of the common MMLU task (originally in the English language) into 14 different languages.

Title: OpenAI o1 System Card

Homepage: https://openai.com/index/openai-o1-system-card/

Technical Report: https://assets.ctfassets.net/kftzwdyauwt9/67qJD51Aur3eIc96iOfeOP/71551c3d223cd97e591aa89567306912/o1_system_card.pdf

[Measuring Massive Multitask Language Understanding](https://arxiv.org/pdf/2009.03300) by Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (ICLR 2021).

### Groups and Tasks

#### Groups

 * `openai_mmmlu_cot`  # supergroup of the following
   * `openai_mmmlu_cot_ar_xy`
   * `openai_mmmlu_cot_bn_bd`
   * `openai_mmmlu_cot_de_de`
   * `openai_mmmlu_cot_es_la`
   * `openai_mmmlu_cot_fr_fr`
   * `openai_mmmlu_cot_hi_in`
   * `openai_mmmlu_cot_id_id`
   * `openai_mmmlu_cot_it_it`
   * `openai_mmmlu_cot_ja_jp`
   * `openai_mmmlu_cot_ko_kr`
   * `openai_mmmlu_cot_pt_br`
   * `openai_mmmlu_cot_sw_ke`
   * `openai_mmmlu_cot_yo_ng`
   * `openai_mmmlu_cot_zh_ch`
 * `openai_mmmlu_default`
   * `openai_mmmlu_default_ar_xy`
   * `openai_mmmlu_default_bn_bd`
   * `openai_mmmlu_default_de_de`
   * `openai_mmmlu_default_es_la`
   * `openai_mmmlu_default_fr_fr`
   * `openai_mmmlu_default_hi_in`
   * `openai_mmmlu_default_id_id`
   * `openai_mmmlu_default_it_it`
   * `openai_mmmlu_default_ja_jp`
   * `openai_mmmlu_default_ko_kr`
   * `openai_mmmlu_default_pt_br`
   * `openai_mmmlu_default_sw_ke`
   * `openai_mmmlu_default_yo_ng`
   * `openai_mmmlu_default_zh_ch`

#### Tasks

* `openai_mmmlu_<prompt>_<language>_<subject>`: The mmlu translation combined with prompt (one of `default,cot`) language (one of )

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
