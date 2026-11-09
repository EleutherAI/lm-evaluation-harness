# AraDiCE

### Paper

**Title:** AraDiCE: Benchmarks for Dialectal and Cultural Capabilities in LLMs

**Abstract:** Arabic, with its rich diversity of dialects, remains significantly underrepresented in Large Language Models, particularly in dialectal variations. We address this gap by introducing seven synthetic datasets in dialects alongside Modern Standard Arabic (MSA), created using Machine Translation (MT) combined with human post-editing. We present AraDiCE, a benchmark for Arabic Dialect and Cultural Evaluation. We evaluate LLMs on dialect comprehension and generation, focusing specifically on low-resource Arabic dialects. Additionally, we introduce the first-ever fine-grained benchmark designed to evaluate cultural awareness across the Gulf, Egypt, and Levant regions, providing a novel dimension to LLM evaluation. Our findings demonstrate that while Arabic-specific models like Jais and AceGPT outperform multilingual models on dialectal tasks, significant challenges persist in dialect identification, generation, and translation. This work contributes ~45K post-edited samples, a cultural benchmark, and highlights the importance of tailored training to improve LLM performance in capturing the nuances of diverse Arabic dialects and cultural contexts. We will release the dialectal translation models and benchmarks curated in this study.

**Homepage:**
https://huggingface.co/datasets/QCRI/AraDiCE



### Citation

```
@article{mousi2024aradicebenchmarksdialectalcultural,
      title={{AraDiCE}: Benchmarks for Dialectal and Cultural Capabilities in LLMs},
      author={Basel Mousi and Nadir Durrani and Fatema Ahmad and Md. Arid Hasan and Maram Hasanain and Tameem Kabbani and Fahim Dalvi and Shammur Absar Chowdhury and Firoj Alam},
      year={2024},
      publisher={arXiv:2409.11404},
      url={https://arxiv.org/abs/2409.11404},
}
```

### Groups, Tags, and Tasks

#### Groups

* `AraDiCE`: Overall results for all tasks associated with different datasets.


#### Tasks

* `aradice`: Overall results for all tasks associated with different datasets.
* `arabicmmlu`: TODO


### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
