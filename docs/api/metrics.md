# Metrics

The metrics system provides per-sample scoring, corpus-level aggregation, and reduction functions for handling repeated evaluations.

[:material-github: Source](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/api/metrics){ .md-button }

## Built-in Metrics

### Loglikelihood Metrics

::: lm_eval.api.metrics.acc
    options:
      show_root_heading: true

::: lm_eval.api.metrics.acc_norm
    options:
      show_root_heading: true

::: lm_eval.api.metrics.acc_bytes
    options:
      show_root_heading: true

::: lm_eval.api.metrics.acc_mutual_info_fn
    options:
      show_root_heading: true

::: lm_eval.api.metrics.exact_match_mc
    options:
      show_root_heading: true

::: lm_eval.api.metrics.bpb
    options:
      show_root_heading: true

::: lm_eval.api.metrics.logprob_fn
    options:
      show_root_heading: true

::: lm_eval.api.metrics.brier_score
    options:
      show_root_heading: true

### Generation Metrics

::: lm_eval.api.metrics.exact_match_fn
    options:
      show_root_heading: true

## Aggregation Functions

::: lm_eval.api.metrics.mean
    options:
      show_root_heading: true

::: lm_eval.api.metrics.median
    options:
      show_root_heading: true

::: lm_eval.api.metrics.nanmean
    options:
      show_root_heading: true

::: lm_eval.api.metrics.weighted_mean
    options:
      show_root_heading: true

::: lm_eval.api.metrics.perplexity
    options:
      show_root_heading: true

::: lm_eval.api.metrics.weighted_perplexity
    options:
      show_root_heading: true

::: lm_eval.api.metrics.bits_per_byte
    options:
      show_root_heading: true

### Corpus-Level Metrics

Metrics that must operate across the entire corpus rather than per-sample.

::: lm_eval.api.metrics.Perplexity
    options:
      show_root_heading: true

::: lm_eval.api.metrics.WordPerplexity
    options:
      show_root_heading: true

::: lm_eval.api.metrics.BytePerplexity
    options:
      show_root_heading: true

::: lm_eval.api.metrics.BitsPerByte
    options:
      show_root_heading: true

::: lm_eval.api.metrics.Bleu
    options:
      show_root_heading: true

::: lm_eval.api.metrics.Chrf
    options:
      show_root_heading: true

::: lm_eval.api.metrics.Ter
    options:
      show_root_heading: true

::: lm_eval.api.metrics.F1
    options:
      show_root_heading: true

::: lm_eval.api.metrics.MCC
    options:
      show_root_heading: true

## Stderr Functions

::: lm_eval.api.metrics.stderr_for_metric
    options:
      show_root_heading: true

::: lm_eval.api.metrics.bootstrap_stderr
    options:
      show_root_heading: true

::: lm_eval.api.metrics.mean_stderr
    options:
      show_root_heading: true

## Types

The core metric wrapper. Each metric defines a per-sample function, an aggregation strategy, and optionally a reduction for repeated samples.

::: lm_eval.api.metrics.Metric
    options:
      show_root_heading: true

::: lm_eval.api.metrics.MetricFn
    options:
      show_root_heading: true

::: lm_eval.api.metrics.AggregationFn
    options:
      show_root_heading: true

::: lm_eval.api.metrics.ReductionFn
    options:
      show_root_heading: true

::: lm_eval.api.metrics.CorpusMetric
    options:
      show_root_heading: true

::: lm_eval.api.metrics.LLResults
