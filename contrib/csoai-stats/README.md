# lm-eval-harness: signed, interval-annotated results (coordinate-first PR text)

## What
- Wilson 95% intervals + MDE table on every proportion score in results JSON
  (sample-size honesty: an acc=0.9 at n=20 is [0.70, 0.97], not a headline).
- `_content_hash = sha256(canonical results)` + optional `_signature` field so a
  leaderboard entry round-trips as an evidence unit (verify offline).
- No API break: additive fields only; opt-in signature via a callback.

## Why
Unsigned measurement dies (Open LLM Leaderboard, Mar 2025). Labs' system cards
need scores a third party can recompute and verify — intervals + hash make that
possible without trusting the publisher.

## Coordinate-first
Drafted before any PR; happy to fold into existing `results` serialization or a
`--with-intervals` flag per maintainer preference. One genuine contribution.
