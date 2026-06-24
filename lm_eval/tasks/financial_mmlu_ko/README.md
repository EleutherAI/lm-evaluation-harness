# financial-mmlu-ko

### Paper  
Title: N/A  
Abstract: N/A  
Short description: This benchmark is not based on an existing paper. It is a Korean multiple-choice dataset in the financial domain designed for objective evaluation, where the candidate answers vary per question.  
Homepage: N/A

### Citation
```
@misc{financial_mmlu_ko,
  title = {financial-mmlu-ko: A Korean Financial Multiple-Choice Benchmark},
  note = {Dataset created by allganize on Hugging Face. Questions are collected from public financial sources and verified by experts.},
  year = {2025},
  url = {https://huggingface.co/allganize}
}
```

### Groups, Tags, and Tasks

#### Groups
* **financial**: Multiple-choice tasks in the financial domain.

#### Tags
* **korean**: Tasks and datasets focused on Korean language evaluation.
* **finance**: Evaluation tasks related to the financial domain.
* **multiple_choice**: Although this task uses the generate_until output type due to varying candidate counts, it is fundamentally a multiple-choice problem.

#### Tasks
* **financial_mmlu-ko**: A task that evaluates models on a set of Korean financial multiple-choice questions. Since the number of answer candidates varies per question, the task is implemented using the generate_until paradigm.

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task? *(N/A for this dataset)*
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test? *(N/A)*

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

---

**Additional Information:**

- **Dataset Description:**  
  The **financial-mmlu-ko** dataset is a Korean multiple-choice dataset focused on the financial domain. It comprises two sets of questions:
  - **Curated Questions (104 items):** Collected by crawling and manually verifying questions from public financial sources.
  - **Generated Questions (315 items):** Created using GPT-4 based on finance-related texts and subsequently reviewed by experts.

- **Data Sources:**  
  The questions were collected from:
  * [Korean Wikipedia Finance Category](https://ko.wikipedia.org/wiki/%EB%B6%84%EB%A5%98:%EA%B8%88%EC%9C%B5)
  * [Bank of Korea Economic Research Reports](https://www.bok.or.kr/portal/bbs/P0002454/list.do?menuNo=200431)
  * [경제배움e - 퀴즈로 배우는 시사.경제](https://www.econedu.go.kr/mec/ots/brd/list.do?mnuBaseId=MNU0000286&tplSer=ac73e13e-2d3c-485c-b7fe-a5823b527ead)

- **Implementation Details:**  
  This task uses the **generate_until** output type since the candidate answers vary per question. In the processing of results, a custom `process_results` function (defined in `utils.py`) is used to compare the model’s output against the gold answer by extracting a relevant number from the text.

- **Task Integration:**  
  The task was added to lm-evaluation-harness by [choics2623](https://github.com/choics2623). (Note: In the original Citation, the task integrator is not included.)
