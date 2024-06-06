# LAMBADA

### Paper
The LAMBADA dataset: Word prediction requiring a broad discourse context
https://arxiv.org/pdf/1606.06031.pdf

LAMBADA is a dataset to evaluate the capabilities of computational models for text
understanding by means of a word prediction task. LAMBADA is a collection of narrative
passages sharing the characteristic that human subjects are able to guess their last
word if they are exposed to the whole passage, but not if they only see the last
sentence preceding the target word. To succeed on LAMBADA, computational models
cannot simply rely on local context, but must be able to keep track of information
in the broader discourse.

Homepage: https://zenodo.org/record/2630551#.X4Xzn5NKjUI

### Citation

@misc{
    author={Paperno, Denis and Kruszewski, Germán and Lazaridou, Angeliki and Pham, Quan Ngoc and Bernardi, Raffaella and Pezzelle, Sandro and Baroni, Marco and Boleda, Gemma and Fernández, Raquel},
    title={The LAMBADA dataset},
    DOI={10.5281/zenodo.2630551},
    publisher={Zenodo},
    year={2016},
    month={Aug}
}

@article{bellagente2024stable,
  title={Stable LM 2 1.6 B Technical Report},
  author={Bellagente, Marco and Tow, Jonathan and Mahan, Dakota and Phung, Duy and Zhuravinskyi, Maksym and Adithyan, Reshinth and Baicoianu, James and Brooks, Ben and Cooper, Nathan and Datta, Ashish and others},
  journal={arXiv preprint arXiv:2402.17834},
  year={2024}
}

### Groups and Tasks

#### Groups

* `lambada_multilingual_stablelm`: Evaluates all `lambada_mt_stablelm_X` tasks

#### Tasks

* `lambada_mt_stablelm_{en, fr, de, it, es}`: Machine-translated versions of OpenAI's Lambada variant as reported in "Stable LM 2 1.6 B Technical Report" (Bellagente et. al.).

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
(This task is novel to the Evaluation Harness, and has been checked against v0.3.0 of the harness.)


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
