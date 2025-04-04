# JSONSchema Bench

## Tasks

- `jsonschema_bench_easy`, corresponding to the `github_easy` split of the original paper
- `jsonschema_bench_medium`, corresponding to the `github_medium` split of the original paper
- `jsonschema_bench_hard`, corresponding to the `github_hard` split of the original paper

Use `jsonschema_bench` tag to run all three tasks.

## Metrics

The JSONSchema Bench tasks are evaluated using the following two metrics:
- `json_validity`: This metric checks whether the generated output is valid JSON. It is a binary metric, where 1 indicates valid JSON and 0 indicates invalid JSON. We use `json` package to check the validity of the generated output.
- `schema_compliance`: This metric checks whether the generated output complies with the provided JSON schema. It is also a binary metric, where 1 indicates compliance and 0 indicates non-compliance. We use the `jsonschema` package to check the compliance of the generated output with the provided JSON schema.

## Dependencies

The JSONSchema Bench tasks require the `jsonschema` library to be installed. You can install it using pip:
```bash
pip install jsonschema\[format\]
```

The `format` extra is required to support the `format` keyword in JSON Schema, which is used in the tasks.

##  Sequence Length
The `easy` task requires a context window of 2K tokens, the `medium` task requires a context window of 3K tokens, and the `hard` task requires a context window of 10K tokens ( the exact number will vary depending on the tokenizer used, but 10K tokens is a good estimate).

If you don't have enough memory to run the `hard` task, you can use the `--max_length` flag to reduce the context window size but this will truncate the schema and will lead to lower performance.


## Usage

Here is an example of how to run 10 instances of the `jsonschema_bench_easy` task :
```bash
lm_eval \
    --model hf --gen_kwargs max_new_tokens=1024 \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,parallelize=True\
    --tasks jsonschema_bench_medium \
    --batch_size auto \
    --limit 10 \
    --apply_chat_template \
    --fewshot_as_multiturn
```

The expected results is
```
|         Tasks         |Version|Filter|n-shot|     Metric      |   |Value|   |Stderr|
|-----------------------|------:|------|-----:|-----------------|---|----:|---|-----:|
|jsonschema_bench_medium|    0.1|none  |     2|json_validity    |↑  |  1.0|±  |0.0000|
|                       |       |none  |     2|schema_compliance|↑  |  0.2|±  |0.1333|
```

## Dataset

Available at [HF hub](https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench)

## Leaderboard

We provide a [leaderboard](https://github.com/epfl-dlab/jsonschemabench-leaderboard) to track the progress of LLMs on the JSONSchema Bench tasks.
We welcome contributions to the leaderboard via pull requests.


## Paper

JGenerating Structured Outputs from Language Models: Benchmark and Studies[https://arxiv.org/abs/2501.10868]

Homepage: https://github.com/guidance-ai/jsonschemabench


## Citation
```
@misc{geng2025jsonschemabench,
      title={Generating Structured Outputs from Language Models: Benchmark and Studies},
      author={Saibo Geng and Hudson Cooper and Michał Moskal and Samuel Jenkins and Julian Berman and Nathan Ranchin and Robert West and Eric Horvitz and Harsha Nori},
      year={2025},
      eprint={2501.10868},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.10868},
}
```



### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
