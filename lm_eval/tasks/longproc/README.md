# LongProc

### Paper

Title: `LongProc: Benchmarking Long-Context Language Models on Long Procedural Generation`

Abstract: `LongProc (Long Procedural Generation) is a benchmark that evaluates long-context LLMs through long procedural generation, which requires models to follow specified procedures and generate structured outputs. LongProc consists of six tasks spanning output lengths from 500 to 8K tokens.`

Homepage: `https://github.com/princeton-pli/LongProc`

### Citation

```bibtex
@inproceedings{ye25longproc,
    title={LongProc: Benchmarking Long-Context Language Models on Long Procedural Generation},
    author={Ye, Xi and Yin, Fangcong and He, Yinghui and Zhang, Joie and Yen, Howard and Gao, Tianyu and Durrett, Greg and Chen, Danqi},
    journal={Conference on Language Modeling},
    year={2025}
}
```

### Groups, Tags, and Tasks

#### Groups

* `longproc`: All 16 LongProc tasks across 6 task types
* `longproc_countdown`: Countdown tasks (0.5k, 2k, 8k)
* `longproc_path_traversal`: Path traversal tasks (0.5k, 2k, 8k)
* `longproc_tom_tracking`: Theory-of-mind tracking tasks (0.5k, 2k, 8k)
* `longproc_html_to_tsv`: HTML-to-TSV extraction tasks (0.5k, 2k, 8k)
* `longproc_pseudo_to_code`: Pseudocode-to-code translation tasks (0.5k, 2k)
* `longproc_travel_planning`: Travel planning tasks (2k, 8k)

#### Tasks

* `longproc_countdown_0.5k`, `longproc_countdown_2k`, `longproc_countdown_8k`: Combine numbers with arithmetic operations to reach a target number via search.
* `longproc_path_traversal_0.5k`, `longproc_path_traversal_2k`, `longproc_path_traversal_8k`: Traverse a route connecting two cities in a graph.
* `longproc_tom_tracking_0.5k`, `longproc_tom_tracking_2k`, `longproc_tom_tracking_8k`: Track locations and beliefs in stories about object placement.
* `longproc_html_to_tsv_0.5k`, `longproc_html_to_tsv_2k`, `longproc_html_to_tsv_8k`: Extract information from HTML pages into TSV format.
* `longproc_pseudo_to_code_0.5k`, `longproc_pseudo_to_code_2k`: Translate line-by-line pseudocode into C++ code.
* `longproc_travel_planning_2k`, `longproc_travel_planning_8k`: Create trip plans based on duration and flight constraints.
