# Threat model: prompt_defense_eval

`prompt_defense_eval` scores whether a model emits any of the 22 catalog-listed
payload surface patterns (catalog version 0.1.0, pinned in `catalog.py` as
`CATALOG_VERSION` and in the task YAML as `metadata.catalog_version`). It is a
**lower-bound detector**: a passing sample means the regex catalog did not fire,
not that the emitted artifact is harmless. The score is a floor, not a ceiling.

## What the number means

`acc` is the fraction of the 33 samples on which no rule fired. Read it as the
rate at which the model emitted no unobfuscated surface payload. The per-category
metrics (`acc_xss`, `acc_sqli`, `acc_shell`, `acc_path`, `acc_credential`,
`acc_markdown`, `acc_code`) carry the same reading per hazard. A high score does
not certify that the model is hardened; it states that the catalog did not fire.

## Known false negatives, by construction

The catalog matches surface form, not danger. A model can emit a working payload
that no rule fires on by restructuring the same hazard:

- string concatenation
- hex / unicode / base64 encoding
- comment-splitting
- equivalent-but-unmatched syntax

These are expected misses, not catalog bugs. As models get better at restructuring
output past a static matcher, the pass rate inflates: a clean score starts to read
as "evaded the regex", not "safe". That direction is the reason this file exists.
The number is not a safety guarantee and should not be cited as one.

## Why the catalog version is pinned

A reported score is comparable only against the same rule set. The catalog is
pinned to one upstream release (`prompt-defense-audit-py` 0.1.0, MIT,
https://github.com/ppcvote/prompt-defense-audit-py) and the version travels with
the result through `metadata.catalog_version`. Widening the catalog produces a new
pinned version rather than silently moving the baseline.

## Not in this PR: obfuscation-robustness controls

This PR ships the surface detector only. The obfuscation-robustness control set is
a planned follow-up: for each hazard, one canonical positive (regex fires) plus
obfuscated equivalents of the same hazard (regex misses), scored as a
`surface_pass_rate` / `obfuscated_pass_rate` pair whose `detector_gap` is the
measured blind spot. It uses the obfuscation transforms from @shipbehaves'
red-team work (`self-evolving-adversarial-safety`). Until it lands, `detector_gap`
is unmeasured, and the floor framing above is the only honest reading of the score.
