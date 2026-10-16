# serbeval

### Paper

No paper. serbeval is a native (not translated) benchmark of Serbian grammar, morphology
and orthography: the things a translated benchmark cannot carry over.

Homepage: https://github.com/djolex999/serbeval

Hugging Face: https://huggingface.co/datasets/djolex999/serbeval (official set v1, 355 items)

It covers case government, verbal aspect, the aorist and imperfect, sibilarization and
palatalization, numeral + adjective + noun agreement, the ijekavian jat reflex, writing
words together or apart, hyphenated compounds (polusloženice), prefixoids, compound
adjectives, č/ć, and Cyrillic/Latin consistency.

Every item carries a verification status (`MANIFEST.json` in the dataset):
- `human` (63): checked by a native speaker;
- `consensus` (214): fewer than two of four strong models disagreed with the gold;
- `derived` (78): a correction item built mechanically from an official item.

Gold answers come from hand-written items and from the srLex lexicon, using only forms
attested in the corpus. Items where Serbian authorities disagree (Pravopis 2010 vs Klajn,
e.g. *foto-aparat* / *fotoaparat*) were dropped rather than pinned to one side.

### Citation

```bibtex
@misc{serbeval2026,
  title  = {serbeval: a Serbian-language benchmark for large language models},
  author = {Markovic, Djordje},
  note   = {GitHub: @djolex999},
  year   = {2026},
  url    = {https://github.com/djolex999/serbeval}
}
```

### Groups and Tasks

#### Groups

* `serbeval`: all 17 tasks. Its `exact_match` is the unweighted mean over the 17 tasks.
* `serbeval_mcq`: `serbeval_grammar`, `serbeval_idioms`, `serbeval_cyrillic_agreement`
* `serbeval_cloze`: `serbeval_grammar_cloze`
* `serbeval_pravopis`: `serbeval_pravopis_{prefiksoidi, poluslozenice, slozeni_pridevi, sastavljeno_rastavljeno}`
* `serbeval_probe`: `serbeval_probe_{jat, aorist_imperfekat, sibilarizacija, broj_pridev_imenica, sastavljeno_rastavljeno, poluslozenice}`
* `serbeval_correction`: `serbeval_correction_{razmak_crtica, c_c, ispravno}`

The project leaderboard reports the mean of the five group scores instead of the mean of
the 17 tasks. Both are computed from the same per-task results.

#### Tasks

All tasks are zero-shot `generate_until`, so they run on chat APIs without logprobs.
- **MCQ** (`grammar`, `idioms`, `cyrillic_agreement`): four options, scored on the first
  standalone capital A-D. `cyrillic_agreement` also reports `pair_agreement`, the share of
  Cyrillic/Latin pairs answered with the same outcome.
- **Cloze** (`grammar_cloze`, `probe_*`, `pravopis_*`): the model writes one form, given a
  cue in brackets. Case, outer punctuation and script are folded. Serbian diacritics and
  intra-word hyphens are not (`kuca` != `kuća`, `kontakt osoba` != `kontakt-osoba`).
  `answers` lists every accepted variant.
- **Correction** (`correction_*`): no cue. The model gets a sentence with one spacing or
  hyphen error, one č/ć error, or no error at all (`ispravno`, which measures false
  alarms), and answers with the corrected or unchanged sentence after `Ispravljeno:`.

Every task also reports `unparsed`, the share of responses with nothing readable in them,
so a formatting failure is not mistaken for a language error.

### Results

Zero-shot through OpenRouter, mean of the five group scores, official set v1 (Oct 2026).
Full table of 25 models: https://github.com/djolex999/serbeval/blob/main/results/LEADERBOARD.md

| Model | MCQ | Cloze | Pravopis | Probe | Correction | Mean |
|---|---|---|---|---|---|---|
| claude-opus-5.5 | 100 | 100 | 96 | 97 | 99 | 98 |
| gpt-6-sol | 98 | 100 | 97 | 96 | 98 | 98 |
| gemini-3.8-flash | 100 | 100 | 94 | 97 | 98 | 98 |
| gemma-4-31b-it | 97 | 100 | 77 | 78 | 92 | 89 |
| llama-3.3-70b-instruct | 88 | 94 | 57 | 52 | 75 | 73 |
| llama-3.1-8b-instruct | 47 | 50 | 22 | 20 | 53 | 39 |

These task files were checked against the project's own implementation:
- Re-scoring all 25,603 stored responses from the leaderboard runs gives identical
  per-item scores.
- A fresh run of llama-3.1-8b-instruct through this harness, on the same provider at bf16,
  reproduced every group score above.

**Note on open-weights models served by an API router.** The same model scored 4-10 points
lower per group when the router was free to pick providers, some of which serve quantized
builds. Pin the provider and precision when comparing open models.

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature? (No: this is a new benchmark,
  first released with its own repository rather than a paper.)
  * [ ] Have you referenced the original paper that introduced the task? (n/a, no paper)
  * [x] If yes, does the original paper provide a reference implementation? If so, have
    you checked against the reference implementation and documented how to run such a
    test? (The project repository is the reference implementation. These task files
    re-score its 25,603 stored responses identically and reproduce a reference run; see
    Results.)

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

- Oct 2026: version 1.0, first release (official set v1, 355 items).
