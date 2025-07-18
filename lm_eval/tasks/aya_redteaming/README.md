# Aya Redteaming

The [Aya Red-teaming dataset](https://huggingface.co/datasets/CohereLabs/aya_redteaming) is a human-annotated multilingual red-teaming dataset consisting of harmful prompts in 8 languages across 9 different categories of harm with explicit labels for "global" and "local" harm.

## Usage

For all languages:
```bash
python -m lm_eval --model <model_name> --tasks aya_redteaming
```

For one specific language, e.g. Spanish:
```bash
python -m lm_eval --model <model_name> --tasks aya_redteaming_spanish
```

## Citation

```
@misc{aakanksha2024multilingualalignmentprismaligning,
      title={The Multilingual Alignment Prism: Aligning Global and Local Preferences to Reduce Harm}, 
      author={Aakanksha and Arash Ahmadian and Beyza Ermis and Seraphina Goldfarb-Tarrant and Julia Kreutzer and Marzieh Fadaee and Sara Hooker},
      year={2024},
      eprint={2406.18682},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.18682}, 
}
``` 