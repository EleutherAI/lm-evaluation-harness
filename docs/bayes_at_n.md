# Bayes@N for repeated exact-match evaluation

`bayes_at_n` reports the posterior mean of average correctness across a task's documents. `bayes_at_n_std` reports the posterior standard deviation of that same average. Both metrics use the binary, uniform-prior version of [Bayes@N](https://arxiv.org/abs/2510.04265), with an independent `Beta(1, 1)` prior for each document.

For a document with `s` correct responses out of `n` independent samples, the posterior is `Beta(s + 1, n - s + 1)`. With `M` documents, the corpus mean is the average of their posterior means. The corpus standard deviation is `sqrt(sum(document_variances)) / M`. The implementation keeps each document's outcomes separate, including when gathering results from multiple ranks.

## Configure a task

Add these fields to a `generate_until` task whose target can be graded with exact match:

```yaml
output_type: generate_until
repeats: 8
generation_kwargs:
  do_sample: true
  temperature: 0.8
  until:
    - "\n"
filter_list:
  - name: eight_samples
    filter:
      - function: take_first_k
        k: 8
metric_list:
  - metric: bayes_at_n
    n: 8
  - metric: bayes_at_n_std
    n: 8
```

Keep the task's dataset, prompt and target fields. Run the task with the usual `lm-eval` CLI or Python evaluation API. Sampling options depend on the model backend; configure it to draw independent responses to each fixed prompt. The response cache is bypassed for `generate_until` requests with `do_sample: true`.

Set `n` on both metrics to the number of retained responses. It must be a positive integer. The metric rejects a missing response, a shorter row or a bare string returned by `take_first` or majority selection. If earlier filters extract answers, they must preserve every sample before `take_first_k` selects the row. The existing default filter discards repeats, so the explicit filter above is required.

Both metrics accept the usual `exact_match` options: `regexes_to_ignore`, `ignore_case`, `ignore_punctuation` and `ignore_numbers`. Set the same options on both metric entries. If a document has several accepted targets, each sampled response is correct when it matches any target. The targets remain answer alternatives, with one binary outcome per sampled response.

## Read the results

The result keys are `bayes_at_n,eight_samples` and `bayes_at_n_std,eight_samples`, on a 0–1 scale. For two documents with outcomes `[[0, 0], [1, 1]]`, the posterior mean is `0.5` and the posterior standard deviation is `sqrt(3 / 160)`, about `0.13693`. By comparison, `[[0, 1], [0, 1]]` has the same mean but posterior standard deviation `sqrt(1 / 40)`, about `0.15811`.

The standard deviation measures posterior uncertainty about the fixed task's average latent correctness. It is separate from bootstrap sampling stderr and from pass@k's probability of at least one success. The harness's `_stderr` fields remain `N/A` for these aggregations, even when bootstrap iterations are enabled. The standard deviation has no higher-is-better direction.

These metrics support per-task corpus results, including evaluations containing several tasks. Do not add either field to a group's `aggregate_metric_list`; this raises an error. Group summaries currently lack the information needed to establish a common repeated-sampling experiment and combine its posterior uncertainty. Each task's own Bayes@N results remain available when a group aggregates other metrics.

This integration supports binary exact-match outcomes, a fixed number of samples per document and no prior observations. It does not support categorical rubrics, fractional rewards, adaptive sample counts or credible intervals. Custom `process_results` functions can use the registered aggregations with complete binary rows of equal length; they must retain every evaluated document and supply the same rows for both metrics.

The formulas follow Algorithm 1 and Appendix A of Hariri et al., *Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation*. The [Scorio reference implementation](https://github.com/mohsenhariri/scorio/blob/7278f677c30396d99a0454bcbb08f5a9a53c4efb/scorio/eval/bayes.py) also supports categorical outcomes and prior observations. No Scorio dependency is needed for this binary implementation.
