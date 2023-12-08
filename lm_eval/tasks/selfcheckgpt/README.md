# SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models

In order to run selfcheckgpt evaluation, these dependencies should be installed.
```
pip install spacy
pip install selfcheckgpt
python -m spacy download en
```

selfcheckgpt support different evaluation methods including: `SelfCheckNgram`, `SelfCheckBERTScore`, `SelfCheckMQAG` and `SelfCheckNLI`.
The default evaluation method in llm-eval-harness is `SelfCheckNgram`. You can change the evaluation method by changing the environment variable
```
export SELFCHECKGPTTYPE=SelfCheckNgram
```
For `SelfCheckBERTScore`, `SelfCheckMQAG` and `SelfCheckNLI` evaluation method which will also run some huggingface models, You can change the running device of the selfcheckgpt to GPU by setting enviroment device:
```
export SELFCHECKGPTDEVICE=cuda
```

## Citation

```
@misc{manakul2023selfcheckgpt,
      title={SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models},
      author={Potsawee Manakul and Adian Liusie and Mark J. F. Gales},
      year={2023},
      eprint={2303.08896},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```