# FDA

### Paper

Title: Language Models Enable Simple Systems For
Generating Structured Views Of Heterogenous Data
Lakes

Abstract: A long standing goal of the data management community is to develop general, automated systems
that ingest semi-structured documents and output queryable tables without human effort or domain
specific customization. Given the sheer variety of potential documents, state-of-the art systems make
simplifying assumptions and use domain specific training. In this work, we ask whether we can
maintain generality by using large language models (LLMs). LLMs, which are pretrained on broad
data, can perform diverse downstream tasks simply conditioned on natural language task descriptions.
We propose and evaluate EVAPORATE, a simple, prototype system powered by LLMs. We identify
two fundamentally different strategies for implementing this system: prompt the LLM to directly
extract values from documents or prompt the LLM to synthesize code that performs the extraction.
Our evaluations show a cost-quality tradeoff between these two approaches. Code synthesis is cheap,
but far less accurate than directly processing each document with the LLM. To improve quality while
maintaining low cost, we propose an extended code synthesis implementation, EVAPORATE-CODE+,
which achieves better quality than direct extraction. Our key insight is to generate many candidate
functions and ensemble their extractions using weak supervision. EVAPORATE-CODE+ not only
outperforms the state-of-the art systems, but does so using a sublinear pass over the documents with
the LLM. This equates to a 110× reduction in the number of tokens the LLM needs to process,
averaged across 16 real-world evaluation settings of 10k documents each.


A task for LMs to perform Information Extraction, as implemented by Based.

Homepage: https://github.com/HazyResearch/based-evaluation-harness


Description:
> FDA (Information Extraction). The task is to extract key-value pairs from a set of PDFs scraped from the FDA website. We use the dataset and labels collected in Arora et al. 2023. We break apart the documents into chunks of 1,920 tokens. For every key-value pair that appears in the chunk, we create a zero-shot prompt using the simple prompt template: {chunk} \n {key}: We allow the model to generate a fixed number of tokens after the prompt and check (with case insensitivity) if the value is contained within the generation. We report accuracy, the fraction of prompts for which the generation contains the value.



### Citation

```
@misc{arora2024simple,
      title={Simple linear attention language models balance the recall-throughput tradeoff},
      author={Simran Arora and Sabri Eyuboglu and Michael Zhang and Aman Timalsina and Silas Alberti and Dylan Zinsley and James Zou and Atri Rudra and Christopher Ré},
      year={2024},
      eprint={2402.18668},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{arora2023language,
      title={Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes},
      author={Simran Arora and Brandon Yang and Sabri Eyuboglu and Avanika Narayan and Andrew Hojel and Immanuel Trummer and Christopher Ré},
      year={2023},
      eprint={2304.09433},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```

### Groups and Tasks

#### Tasks

* `fda`: the FDA task as implemented in the paper "Simple linear attention language models balance the recall-throughput tradeoff". Designed for zero-shot evaluation of small LMs.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
