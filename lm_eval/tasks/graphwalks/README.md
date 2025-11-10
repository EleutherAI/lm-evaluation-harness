# GraphWalks: a multi hop reasoning long context benchmark
In Graphwalks, the model is given a graph represented by its edge list and asked to perform an operation.

### Dataset

HuggingFace: https://huggingface.co/datasets/openai/graphwalks

### Groups and Tasks

#### Groups

* `graphwalks`: Run both `graphwalks_128k` and `graphwalks_1M`

#### Tasks

* `graphwalks_128k`: Up to 128k context length
* `graphwalks_1M`: Between 256k-1M context length

> [!NOTE]
> Please note that `max_gen_toks` is set to `16384`, but non-reasoning models do not need this many tokens.


### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
