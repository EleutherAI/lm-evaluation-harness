# Physics GRE

### Source

The Physics GRE dataset is released as a part of the [Inflection Benchmark](https://github.com/InflectionAI/Inflection-Benchmarks). It contains four processed Physics GRE exams, a common graduate school entrance exam for physics students. Each question in this dataset has five options and only one answer is correct.



<!-- Title: `paper titles goes here` -->

<!-- Abstract: `link to paper PDF or arXiv abstract goes here` -->

<!-- `Short description of paper / benchmark goes here:` -->

<!-- Homepage: `homepage to the benchmark's website goes here, if applicable` -->


### Tasks

<!-- #### Groups

* `group_name`: `Short description` -->

#### Task List

* `physics_gre`: `Exam GR8677`
* `physics_gre_additional`: `GRE exams (GR9277, GR9677, and GR0177)`
* `physics_gre_all`: `GRE exams (GR9277, GR9677, GR0177, and GR8677)`

#### Evaluation Setup and Metrics

We evaluate `Raw Score` and `Percentile` in `score-first`, `maj@8`, and `maj@32` setups with generation `temperature=0.2`. We include only questions without an image in our scoring.


#### Exam Scoring Details
For the Physics GRE, each correct answer is worth 1 point and each incorrect answer results in a -0.25 reduction.
To compute the score, we make the following assumption:
```
Raw_Score = Percentage_Correct - 0.25 * (1 - Percentage_Correct)
```
where `Percentage_Correct` is computed purely on questions without images. For simplicity, we do not use heuristics to allow the model not to answer.

| Raw Score    | Percentile |
| -----------: | ---------: |
| 81 &ndash; 100       | 98         |
| 77 &ndash; 80        | 97         |
| 75 &ndash; 76        | 96         |
| 72 &ndash; 74        | 95         |
| 71           | 94         |
| 69 &ndash; 70        | 93         |
| 67 &ndash; 68        | 92         |
| 65 &ndash; 66        | 91         |
| 64           | 90         |
| 63           | 89         |
| 61 &ndash; 62        | 87         |
| 60           | 86         |
| 59           | 85         |
| 57 &ndash; 58        | 84         |
| 56           | 82         |
| 55           | 80         |
| 53 &ndash; 54        | 78         |
| 52           | 77         |
| 51           | 75         |
| 49 &ndash; 50        | 72         |
| 48           | 70         |
| 47           | 69         |
| 45 &ndash; 46        | 66         |
| 44           | 64         |
| 43           | 62         |
| 41 &ndash; 42        | 59         |
| 40           | 57         |
| 39           | 54         |
| 37 &ndash; 38        | 52         |
| 36           | 48         |
| 35           | 46         |
| 33 &ndash; 34        | 43         |
| 32           | 41         |
| 30 &ndash; 31        | 38         |
| 29           | 35         |
| 28           | 32         |
| 26 &ndash; 27        | 30         |
| 25           | 27         |
| 24           | 25         |
| 22 &ndash; 23        | 22         |
| 21           | 20         |
| 20           | 18         |
| 18 &ndash; 19        | 16         |
| 17           | 14         |
| 16           | 12         |
| 14 &ndash; 15        | 10         |
| 13           | 9          |
| 12           | 8          |
| 10 &ndash; 11        | 6          |
| 9            | 5          |
| 8            | 4          |
| 6 &ndash; 7          | 3          |
| 5            | 2          |
| 1 &ndash; 4          | 1          |
| 0            | 0          |

### Reference Performance of Models

Model: `mistralai/Mistral-7B-Instruct-v0.2`

|        Tasks         |Version|  Filter   |n-shot|  Metric  |Value|
|----------------------|------:|-----------|-----:|----------|----:|
|physics_gre_all       |      1|score-first|     0|raw_score |   18|
|                      |       |score-first|     0|percentile|   16|
|                      |       |maj@8      |     0|raw_score |   18|
|                      |       |maj@8      |     0|percentile|   16|
|                      |       |maj@32     |     0|raw_score |   17|
|                      |       |maj@32     |     0|percentile|   14|
|physics_gre_additional|      1|score-first|     0|raw_score |   13|
|                      |       |score-first|     0|percentile|    9|
|                      |       |maj@8      |     0|raw_score |   16|
|                      |       |maj@8      |     0|percentile|   12|
|                      |       |maj@32     |     0|raw_score |   15|
|                      |       |maj@32     |     0|percentile|   10|
|physics_gre           |      1|score-first|     0|raw_score |   23|
|                      |       |score-first|     0|percentile|   22|
|                      |       |maj@8      |     0|raw_score |   25|
|                      |       |maj@8      |     0|percentile|   27|
|                      |       |maj@32     |     0|raw_score |   23|
|                      |       |maj@32     |     0|percentile|   22|

Model: `mistralai/Mixtral-8x7B-Instruct-v0.1`, 4-bit QLoRA

|        Tasks         |Version|  Filter   |n-shot|  Metric  |Value|
|----------------------|------:|-----------|-----:|----------|----:|
|physics_gre_all       |      1|score-first|     0|raw_score |   32|
|                      |       |score-first|     0|percentile|   41|
|                      |       |maj@8      |     0|raw_score |   35|
|                      |       |maj@8      |     0|percentile|   46|
|                      |       |maj@32     |     0|raw_score |   34|
|                      |       |maj@32     |     0|percentile|   43|
|physics_gre_additional|      1|score-first|     0|raw_score |   28|
|                      |       |score-first|     0|percentile|   32|
|                      |       |maj@8      |     0|raw_score |   31|
|                      |       |maj@8      |     0|percentile|   38|
|                      |       |maj@32     |     0|raw_score |   32|
|                      |       |maj@32     |     0|percentile|   41|
|physics_gre           |      1|score-first|     0|raw_score |   42|
|                      |       |score-first|     0|percentile|   59|
|                      |       |maj@8      |     0|raw_score |   42|
|                      |       |maj@8      |     0|percentile|   59|
|                      |       |maj@32     |     0|raw_score |   42|
|                      |       |maj@32     |     0|percentile|   59|

#### Reported by Inflection

| Model                 | Percentile |
| ----------------------| ---------: |
| Inflection-2.5 maj@8  | 85         |
| Inflection-2.5 maj@32 | 95         |
| GPT-4 maj@8           | 97         |


### Using this benchmark

```bash
lm_eval --model hf --model_args pretrained=mistralai/Mistral-7B-Instruct-v0.2 \
	--tasks physics_gre,physics_gre_additional,physics_gre_all \
	--device cuda:0 --batch_size 8 --output_path ./output --log_samples

lm_eval --model hf \
  --model_args pretrained=mistralai/Mixtral-8x7B-Instruct-v0.1,load_in_4bit=True \
  --tasks physics_gre,physics_gre_additional,physics_gre_all \
  --device cuda:0 --batch_size 1 --output_path ./output --log_samples
```

### Citation

```
@misc{Kuttler2024,
  author = {Kuttler, H},
  title = {Public Inflection Benchmarks },
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/InflectionAI/Inflection-Benchmarks}},
  commit = {706d0f22e5dd2c8ac670c472bd98fb2f93af19ca}
}
```

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
