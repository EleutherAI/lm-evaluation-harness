# Titulm Bangla MMLU

This repository contains resources related to **Titulm Bangla MMLU**, a benchmark dataset designed for evaluating Bangla language models. The dataset is used for training, development, and comparative evaluation of language models in the Bangla language.

---

## Overview

**TituLLMs** is a family of Bangla large language models (LLMs) with comprehensive benchmarking designed to advance natural language processing for the Bangla language. The benchmark dataset `Titulm Bangla MMLU` covers multiple-choice questions across a diverse range of topics in Bangla.

This dataset is primarily used to train, validate, and evaluate Bangla language models and compare their performance with other existing models.

For more details, please refer to the original research paper:  
[https://arxiv.org/abs/2502.11187](https://arxiv.org/abs/2502.11187)


---

## Dataset

The `Titulm Bangla MMLU` dataset can be found on Hugging Face:  
[https://huggingface.co/datasets/hishab/titulm-bangla-mmlu](https://huggingface.co/datasets/hishab/titulm-bangla-mmlu)

This dataset was used as a benchmark in the development and evaluation of TituLLMs and related models.

---

## Usage

The dataset is intended for use within the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repository to evaluate and compare the performance of Bangla language models.

---

## Note: The dataset can also be used to evaluate other models

### Other datasets like boolq, openbookqa ... soon to be added
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
