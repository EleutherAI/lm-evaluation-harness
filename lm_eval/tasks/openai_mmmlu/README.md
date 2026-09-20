# OpenAI MMMLU

### Technical Report

The task/dataset contains a professional, human-translation of the common MMLU task (originally in the English language) into 14 different languages.

Title: OpenAI o1 System Card

Homepage: https://openai.com/index/openai-o1-system-card/

Technical Report: https://assets.ctfassets.net/kftzwdyauwt9/67qJD51Aur3eIc96iOfeOP/71551c3d223cd97e591aa89567306912/o1_system_card.pdf

[Measuring Massive Multitask Language Understanding](https://arxiv.org/pdf/2009.03300) by Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (ICLR 2021).

### Groups and Tasks

The `default` variant to the common MMLU-style prompting, output from a `--write-out`:

```bash
[...]

document 0; context prompt (starting on next line):  
The following are multiple choice questions (with answers) about anatomy.

Ermitteln Sie den Grad für die gegebene Felderweiterung Q(sqrt(2), sqrt(3), sqrt(18)) über Q.
A. 0
B. 4
C. 2
D. 6
Antwort:
(end of prompt on previous line)
target string or answer choice index (starting on next line):
B
(end of target on previous line)
```

Note that
 * the `description` is in English, while the question itself is in the target language, and the "Answer: " prefix is in the target language [this last bit was my choice].
 * in the paper, the prompt is [significantly different](https://github.com/openai/simple-evals/blob/2df1a92bbddb8c89fbeb3670e2dd125b10632bca/common.py#L12) and includes COT plus [generous regexps](https://github.com/openai/simple-evals/blob/2df1a92bbddb8c89fbeb3670e2dd125b10632bca/common.py#L29) (filters) to extract the answer. I am of the opinion one should implement a different variant to reproduce those results.
 * split information is not present in the [dataset on hf](https://huggingface.co/datasets/openai/MMMLU), so currently this dataset doesn't support fewshot or decontamination.

#### Groups

 * `openai_mmmlu_default`  # supergroup of the following groups
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

* `openai_mmmlu_default_<language>_<subject>`: The mmlu translation combined with prompt, language and subject from mmlu.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted? Yes, it would be the `default` folder.
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
