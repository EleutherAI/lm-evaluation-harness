# Bangla Benchmark Datasets

This repository contains resources related to **Bangla MMLU**, **Bangla OpenBookQA**, **Bangla BoolQA**, **Bangla PIQA**, **Bangla CommonsenseQA**,  benchmark datasets designed for evaluating Bangla language models. The dataset is used for training, development, and comparative evaluation of language models in the Bangla language.

---

## Overview

**TituLLMs** is a family of Bangla large language models (LLMs) with comprehensive benchmarking designed to advance natural language processing for the Bangla language. The benchmarking datasets covers  questions across a diverse range of topics in Bangla.

These datasets were primarily used to train, validate, and evaluate Bangla language models(**TituLLM**) and compare their performance with other existing models.

For more details, please refer to the original research paper:  
[https://arxiv.org/abs/2502.11187](https://arxiv.org/abs/2502.11187)


---

## Dataset

The  datasets can be found on Hugging Face:  
[https://huggingface.co/datasets/hishab/titulm-bangla-mmlu](https://huggingface.co/datasets/hishab/titulm-bangla-mmlu)

[https://huggingface.co/datasets/hishab/boolq_bn](https://huggingface.co/datasets/hishab/boolq_bn)

[https://huggingface.co/datasets/hishab/openbookqa-bn](https://huggingface.co/datasets/hishab/openbookqa-bn)

[https://huggingface.co/datasets/hishab/piqa-bn](https://huggingface.co/datasets/hishab/piqa-bn)

[https://huggingface.co/datasets/hishab/commonsenseqa-bn](https://huggingface.co/datasets/hishab/commonsenseqa-bn)

These datasets were used as a benchmark in the development and evaluation of TituLLMs.

---

## Usage

The dataset is intended for use within the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repository to evaluate and compare the performance of Bangla language models.
---

## Citation

If you use this dataset or model, please cite the original paper:

```bibtex
@misc{nahin2025titullmsfamilybanglallms,
      title={TituLLMs: A Family of Bangla LLMs with Comprehensive Benchmarking},
      author={Shahriar Kabir Nahin and Rabindra Nath Nandi and Sagor Sarker and Quazi Sarwar Muhtaseem and Md Kowsher and Apu Chandraw Shill and Md Ibrahim and Mehadi Hasan Menon and Tareq Al Muntasir and Firoj Alam},
      year={2025},
      eprint={2502.11187},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.11187},
}
