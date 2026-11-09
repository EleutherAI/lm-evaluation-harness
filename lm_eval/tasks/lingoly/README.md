# LingOly


### Paper

Title: `LINGOLY: A Benchmark of Olympiad-Level Linguistic Reasoning Puzzles in Low-Resource and Extinct Languages`

Abstract: `https://arxiv.org/abs/2406.06196`

`In this paper, we present the LingOly benchmark, a novel benchmark for advanced reasoning abilities in large language models. Using challenging Linguistic Olympiad puzzles, we evaluate (i) capabilities for in-context identification and generalisation of linguistic patterns in very low-resource or extinct languages, and (ii) abilities to follow complex task instructions. The LingOly benchmark covers more than 90 mostly low-resource languages, minimising issues of data contamination, and contains 1,133 problems across 6 formats and 5 levels of human difficulty. We assess performance with both direct accuracy and comparison to a no-context baseline to penalise memorisation. Scores from 11 state-of-the-art LLMs demonstrate the benchmark to be challenging, and models perform poorly on the higher difficulty problems. On harder problems, even the top model only achieved 38.7% accuracy, 24.7% improvement over the no-context baseline. Large closed models typically outperform open models, and in general, the higher resource the language, the better the scores. These results indicate, in absence of memorisation, true multi-step out-of-domain reasoning remains a challenge for current language models.`

Homepage: `https://github.com/am-bean/lingOly`


### Citation

```
@article{beanLINGOLYBenchmarkOlympiadLevel2024,
  title = {{LINGOLY}: A Benchmark of Olympiad-Level Linguistic Reasoning Puzzles in Low-Resource and Extinct Languages},
  shorttitle = {{LINGOLY}},
  url = {http://arxiv.org/abs/2406.06196},
  author = {Bean, Andrew M. and Hellsten, Simi and Mayne, Harry and Magomere, Jabez and Chi, Ethan A. and Chi, Ryan and Hale, Scott A. and Kirk, Hannah Rose},
  month = jun,
  year = {2024},
  keywords = {Computer Science - Computation and Language}
}
```

### Tasks

* `lingoly`: `runs both _context and _nocontext and computes the difference`
* `lingoly_context`: `exact match of generations to reference answers`
* `lingoly_nocontext`: `exact match of generations to reference answers, but with context removed`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
