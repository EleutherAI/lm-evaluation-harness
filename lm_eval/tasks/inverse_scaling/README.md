# inverse_scaling

### Paper

Title: `Inverse Scaling: When Bigger Isn't Better`

Abstract: `Work on scaling laws has found that large language models (LMs) show predictable improvements to overall loss with increased scale (model size, training data, and compute). Here, we present evidence for the claim that LMs may show inverse scaling, or worse task performance with increased scale, e.g., due to flaws in the training objective and data. We present empirical evidence of inverse scaling on 11 datasets collected by running a public contest, the Inverse Scaling Prize, with a substantial prize pool. Through analysis of the datasets, along with other examples found in the literature, we identify four potential causes of inverse scaling: (i) preference to repeat memorized sequences over following in-context instructions, (ii) imitation of undesirable patterns in the training data, (iii) tasks containing an easy distractor task which LMs could focus on, rather than the harder real task, and (iv) correct but misleading few-shot demonstrations of the task. We release the winning datasets at this https URL to allow for further investigation of inverse scaling. Our tasks have helped drive the discovery of U-shaped and inverted-U scaling trends, where an initial trend reverses, suggesting that scaling trends are less reliable at predicting the behavior of larger-scale models than previously understood. Overall, our results suggest that there are tasks for which increased model scale alone may not lead to progress, and that more careful thought needs to go into the data and objectives for training language models.`

Note: This is not official implementation of inverse scaling prize. Implemented by h-albert-lee with permission from the authors of the paper.

Homepage: https://github.com/inverse-scaling/prize

### Citation

@article{mckenzie2023inverse,
      title={Inverse Scaling: When Bigger Isn't Better},
      author={Ian R. McKenzie and Alexander Lyzhov and Michael Pieler and Alicia Parrish and Aaron Mueller and Ameya Prabhu and Euan McLean and Aaron Kirtland and Alexis Ross and Alisa Liu and Andrew Gritsevskiy and Daniel Wurgaft and Derik Kauffman and Gabriel Recchia and Jiacheng Liu and Joe Cavanagh and Max Weiss and Sicong Huang and The Floating Droid and Tom Tseng and Tomasz Korbak and Xudong Shen and Yuhui Zhang and Zhengping Zhou and Najoung Kim and Samuel R. Bowman and Ethan Perez},
      journal={arXiv preprint arXiv:2306.09479},
      year={2023}
}

### Groups and Tasks

#### Groups

* `inverse_scaling_mc`: all tasks of Inverse Scaling Prize (currently aside from Prompt Injection), matching their implementations on OPT for multiple-choice type classification tasks. **These match the published dataset versions from the prize, which may slightly differ from numbers in the paper (but have been tested for equivalence to the OPT numbers reported at https://huggingface.co/inverse-scaling/opt-1.3b_eval for multiple sizes.**


#### Tasks

- `inverse_scaling_hindsight_neglect_10shot`
- `inverse_scaling_redefine_math`
- `inverse_scaling_quote_repetition`
- `inverse_scaling_neqa`
- `inverse_scaling_winobias_antistereotype`: not an official Inverse Scaling prize winner, but eval results reported on it at https://huggingface.co/inverse-scaling/opt-1.3b_eval .
- `inverse_scaling_into_the_unknown`
- `inverse_scaling_memo_trap`
- `inverse_scaling_modus_tollens`
- `inverse_scaling_pattern_matching_suppression`
- `inverse_scaling_repetitive_algebra`
- `inverse_scaling_sig_figs`


### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
