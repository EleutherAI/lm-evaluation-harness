# Babilong

### Paper

Title: Babilong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack
Abstract: https://arxiv.org/abs/2406.10149

In recent years, the input context sizes of large language models (LLMs) have increased dramatically. However, existing evaluation methods have not kept pace, failing to comprehensively assess the efficiency of models in handling long contexts. To bridge this gap, we introduce the BABILong benchmark, designed to test language models' ability to reason across facts distributed in extremely long documents. BABILong includes a diverse set of 20 reasoning tasks, including fact chaining, simple induction, deduction, counting, and handling lists/sets. These tasks are challenging on their own, and even more demanding when the required facts are scattered across long natural text. Our evaluations show that popular LLMs effectively utilize only 10-20\% of the context and their performance declines sharply with increased reasoning complexity. Among alternatives to in-context reasoning, Retrieval-Augmented Generation methods achieve a modest 60\% accuracy on single-fact question answering, independent of context length. Among context extension methods, the highest performance is demonstrated by recurrent memory transformers after fine-tuning, enabling the processing of lengths up to 50 million tokens. The BABILong benchmark is extendable to any length to support the evaluation of new upcoming models with increased capabilities, and we provide splits up to 10 million token lengths.

Homepage: https://github.com/booydar/babilong

### Citation

```
@article{kuratov2024babilong,
    title={Babilong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack},
    author={Kuratov, Yuri and Bulatov, Aydar and Anokhin, Petr and Rodkin, Ivan and Sorokin, Dmitry and Burtsev, Mikhail},
    journal={arXiv preprint arXiv:2406.10149},
    year={2024}
}
```

### Groups and Tasks

#### Groups

* `babilong`: All Babilong tasks at 0k context length
* `babilong_longctx`: Babilong tasks between qa1-qa5 at context lengths up to 128k


#### Tasks

The benchmark includes 1000 samples of 20 reasoning tasks at various context lengths:

**QA Tasks (qa1-qa20):**
* `babilong_qa1`: Single supporting fact QA
* `babilong_qa2`: Two supporting facts QA
* `babilong_qa3`: Three supporting facts QA
* `babilong_qa4`: Two argument relations
* `babilong_qa5`: Three argument relations
* `babilong_qa6`: Yes/No questions
* `babilong_qa7`: Counting
* `babilong_qa8`: Lists and sets
* `babilong_qa9`: Simple negation
* `babilong_qa10`: Indefinite knowledge
* `babilong_qa11`: Track person through temporal references
* `babilong_qa12`: Conjunction
* `babilong_qa13`: Compound coreference
* `babilong_qa14`: Time reasoning
* `babilong_qa15`: Basic deduction
* `babilong_qa16`: Basic induction
* `babilong_qa17`: Positional reasoning
* `babilong_qa18`: Size reasoning
* `babilong_qa19`: Path finding
* `babilong_qa20`: Motivation deduction

> [!NOTE]
> When using babilong tasks, please note:
> 1. This is the implementation with 1000 samples per length. You can change the dataset path to `RMT-team/babilong` in `common_utils.py` for the dataset with 100 samples per length, which supports context lengths up to 10M tokens.
> 2. Supported lengths are 0k, 1, 2, 4, 8, 16, 32, 64, 128k tokens for tasks qa1-5. Tasks qa6-20 only have a length of 0k.
> 3. The default maximum sequence length is 0k. For calculating metrics of different max seq lengths, specify additional lengths using the metadata parameter:
>   `--metadata '{"max_seq_lengths":"0k,1k,2k,4k,8k,16k,32k,128k"}'`. The config currently only takes one context length at a time. The metadata parameter can also be passed to the TaskManager (metadata: dict).


### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
