# GalicianBench

### Paper

GalicianBench is a benchmark for evaluating language models in Galician tasks. This is, it evaluates the ability of a language model to understand and generate Galician text. GalicianBench offers a combination of pre-existing, open datasets and datasets developed exclusivelly for this benchmark. All the details of GalicianBench will be published in a paper soon.

The new evaluation datasets included in GalicianBench are:
| Task          | Category       | Homepage  |
|:-------------:|:-----:|:-----:|
| Belebele_gl | Reading Comprehension | https://huggingface.co/datasets/proxectonos/belebele_gl |
| GalCoLA | Linguistic Acceptability | https://huggingface.co/datasets/proxectonos/galcola |
| MGSM_ca | Math | https://huggingface.co/datasets/proxectonos/mgsm_gl |
| Parafrases_gl | Paraphrasing | https://huggingface.co/datasets/proxectonos/parafrases_gl |
| PAWS-gl | Paraphrasing | https://huggingface.co/datasets/proxectonos/PAWS-gl |
| OpenBookQA_gl | Question Answering | https://huggingface.co/datasets/proxectonos/openbookqa_gl |
| Summarization_gl | Summarization | https://huggingface.co/datasets/proxectonos/summarization_gl |
| TruthfulQA_gl | Truthfulness | https://huggingface.co/datasets/proxectonos/truthfulqa_gl |
| xnli_gl | NLI | https://huggingface.co/datasets/proxectonos/xnli_gl |
| xstorycloze_gl | Commonsense Reasoning | https://huggingface.co/datasets/proxectonos/xstorycloze_gl |

The datasets included in GalicianBench that have been made public in previous pubications are:

| Task          | Category       | Paper title          | Homepage  |
|:-------------:|:-----:|:-------------:|:-----:|
| FLORES_gl | Translation | [The FLORES-101  Evaluation Benchmark for Low-Resource and Multilingual Machine Translation](https://arxiv.org/abs/2106.03193) | https://huggingface.co/datasets/facebook/flores |


### Citation
Paper for GalicianBench coming soon.

### Groups and Tasks

#### Groups

- `galician_bench`: All tasks included in GalicianBench.
- `flores_gl`: All FLORES translation tasks from or to Galician.


#### Tasks

The following tasks evaluate tasks on GalicianBench dataset using various scoring methods.
  - `belebele_glg_Latn`
  - `flores_gl`
  - `flores_gl-ca`
  - `flores_gl-de`
  - `flores_gl-en`
  - `flores_gl-es`
  - `flores_gl-eu`
  - `flores_gl-fr`
  - `flores_gl-it`
  - `flores_gl-pt`
  - `flores_ca-gl`
  - `flores_de-gl`
  - `flores_en-gl`
  - `flores_es-gl`
  - `flores_eu-gl`
  - `flores_fr-gl`
  - `flores_it-gl`
  - `flores_pt-gl`
  - `galcola`
  - `summarization_gl`
  - `parafrases_gl`
  - `paws_gl`
  - `openbookqa_gl`
  - `mgsm_direct_gl`
  - `truthfulqa_gl`
  - `xnli_gl`
  - `xstorycloze_gl`

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation?
    * [ ] Yes, original implementation contributed by author of the benchmark

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
