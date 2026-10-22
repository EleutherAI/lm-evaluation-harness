# MultiNRC judged replay

`multinrc_judged` scores previously generated final answers using saved local
judge verdicts. The dataset is ScaleAI/MultiNRC, loaded offline from the prepared
run. The native-language test split contains 1,055 questions in French, Spanish
and Chinese. English counterparts are excluded from this experiment.

Prepare the dataset with `python -m lm_eval.judging prepare --config CONFIG`.
Run the configured inference stages with `judge --config CONFIG --stage INDEX`
inside an appropriate GPU allocation. Stage zero generates and freezes answers;
other stages judge those same answers. `report --config CONFIG` reports judge
coverage and comparisons.

For harness replay, set `MULTINRC_RUN` and `MULTINRC_JUDGE`, then select model
`cached-rollouts` with `path=RUN/rollouts.jsonl`, task `multinrc_judged`, zero
few-shot examples and no chat-template transformation. All verdicts must be
valid before replay. Results report the candidate model's judged accuracy.

The rubric compares the question, reference answer and candidate final answer
for semantic correctness. It is an adaptation of the methodology described in
the [MultiNRC paper](https://static.scale.com/uploads/654197dc94d34f66c0f5184e/MultiNRC.pdf),
not a claim of identical leaderboard settings or validated judge reliability.
