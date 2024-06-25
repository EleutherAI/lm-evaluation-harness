# Task-name

### Paper

Title: `Benchmarking Llama2, Mistral, Gemma and GPT for Factuality, Toxicity, Bias and Propensity for Hallucinations`

Abstract: `https://arxiv.org/abs/2404.09785`

This paper introduces fourteen novel datasets for the evaluation of Large Language Models' safety in the context of enterprise tasks. A method was devised to evaluate a model's safety, as determined by its ability to follow instructions and output factual, unbiased, grounded, and appropriate content. In this research, we used OpenAI GPT as point of comparison since it excels at all levels of safety. On the open-source side, for smaller models, Meta Llama2 performs well at factuality and toxicity but has the highest propensity for hallucination. Mistral hallucinates the least but cannot handle toxicity well. It performs well in a dataset mixing several tasks and safety vectors in a narrow vertical domain. Gemma, the newly introduced open-source model based on Google Gemini, is generally balanced but trailing behind. When engaging in back-and-forth conversation (multi-turn prompts), we find that the safety of open-source models degrades significantly. Aside from OpenAI's GPT, Mistral is the only model that still performed well in multi-turn tests. 

Results : `https://innodata-llm-safety.innodata.com/#/`


### Citation

```
@misc{nadeau2024benchmarking,
      title={Benchmarking Llama2, Mistral, Gemma and GPT for Factuality, Toxicity, Bias and Propensity for Hallucinations}, 
      author={David Nadeau and Mike Kroutikov and Karen McNeil and Simon Baribeau},
      year={2024},
      eprint={2404.09785},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```

### Groups and Tasks

#### Groups

* `group_name`: `redlite`

#### Tasks

* `rt-cogensumm`: Tests the model's ability to summarize the prompt without adding any hallucinated information.
* `rt-factcc`: Given a claim about a body of text (both given), assesses if the claim is `Correct` or `Incorrect`
* `rt-frank`: Measures the accuracy of generated summaries.
* `rt-gasm8k-gaia`: Tests the model's ability to reply with `Cannot Answer` rather than hallucinate when it is not given sufficient information to answer.
* `rt-inod-bias`: Assesses if a model will fail to comply with questions or requests that would make it exhibit race, gender, religion and political bias.
* `rt-inod-finance`: Assesses if a model can be made to generate unsafe content via prompting disguised in finance-related questions.
* `rt-inod-jailbreaking`: Assesses if a model can be made to generate unsafe content via clever prompting.

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
