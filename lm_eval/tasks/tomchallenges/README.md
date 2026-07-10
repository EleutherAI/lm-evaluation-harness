## ToMChallenges

This adapter keeps the benchmark's six released prompt families as separate
lm-eval leaves under the `tomchallenges` group:

- `tomchallenges_qa`
- `tomchallenges_comp`
- `tomchallenges_mc`
- `tomchallenges_fb`
- `tomchallenges_tf`
- `tomchallenges_tfr`

It also exposes belief-level breakdown groups for each prompt family:

- `tomchallenges_qa_belief`
- `tomchallenges_comp_belief`
- `tomchallenges_mc_belief`
- `tomchallenges_fb_belief`
- `tomchallenges_tf_belief`
- `tomchallenges_tfr_belief`

Each belief breakdown group expands into six leaves:
`reality`, `anti_reality`, `first_order_a`, `first_order_b`,
`second_order_a`, and `second_order_b`.

The adapter loads the released `Sally-Anne_prompt.csv` and `Smarties_prompt.csv`
files, combines them into a single 360-row evaluation split, and preserves the
original prompt strings for each task family.

The adapter uses `generate_until` for all six prompt families to stay close to
the benchmark's released prompting protocol and generation caps in
`Generate_Models'_Answers.ipynb`.

Scoring notes:
- `mc` preserves the original generative A/B prompt and scores the parsed answer
  letter or option text against the reconstructed gold option.
- The released notebook used a two-token MC cap for API models; Qwen3-8B needs an eight-token cap to emit the requested A/B answer, so the adapter uses `max_gen_toks: 8` for MC.
- `fb` scores whether the normalized gold answer appears in the generated
  fill-in response.
- `tf` and `tfr` parse the A/B true-false judgments and require both judgments
  to match.
- `qa` and `comp` are only an approximation of the paper protocol inside
  lm-eval: the paper used a GPT-4 autograder with few-shot reasoning examples,
  while this adapter uses a deterministic normalized gold-answer containment rule.

Faithfulness notes:
- The faithful reporting surface is the per-leaf accuracy table. The top-level
  `tomchallenges` group is a convenience micro-average across prompt families.
- The adapter does not reproduce the benchmark's GPT-4 autograder for `qa`/`comp`
  or the human grading workflow reported for the other prompt families.
- The adapter keeps the original prompt text, including minor source typos such
  as `Statments:` in the Smarties true-false prompts.

Run:

```bash
lm-eval run --model hf   --model_args pretrained=Qwen/Qwen3-8B,dtype=bfloat16,enable_thinking=false   --apply_chat_template   --tasks tomchallenges   --output_path outputs/tomchallenges_results/Adaptation
```
