# v1.0 Tasks
This list keeps track of which tasks' implementations have been ported to YAML / v2.0 of the Eval Harness.

Boxes should be checked iff tasks are implemented in the refactor and tested for regression. Tasks should be struck through if checked *against original introducing paper* implementation or popularizing implementation. (WIP) Denotes that there exists a PR or person working on this task already.

- [x] Glue
- [x] SuperGlue
- [x] CoQA
- [x] DROP
- [x] ~~Lambada~~
- [x] Lambada (Cloze variants)
- [x] ~~Lambada (Multilingual)~~
- [x] Wikitext
- [x] PiQA
- [x] PROST
- [x] MCTACO
- [x] Pubmed QA
- [x] SciQ
- [x] QASPER
- [x] QA4MRE
- [x] TriviaQA
- [x] AI2 ARC
- [x] LogiQA
- [x] HellaSwag
- [x] SWAG
- [x] OpenBookQA
- [ ] SQuADv2 (Lintang)
- [x] RACE
- [x] HeadQA
- [x] MathQA
- [x] WebQs
- [x] WSC273
- [x] Winogrande
- [x] ANLI
- [x] Hendrycks Ethics (missing some tasks/metrics, see PR 660: <https://github.com/EleutherAI/lm-evaluation-harness/pull/660> for more info)
- [x] TruthfulQA (mc1)
- [x] TruthfulQA (mc2)
- [x] TruthfulQA (gen)
- [x] MuTual
- [ ] Hendrycks Math (Hailey)
- [x] Asdiv
- [ ] GSM8k
- [x] Arithmetic
- [ ] MMMLU (Hailey)
- [x] Translation (WMT) suite
- [x] Unscramble
- [x] ~~Pile (perplexity)~~
- [x] BLiMP
- [x] ToxiGen
- [x] StoryCloze
- [ ] NaturalQs (Hailey)
- [x] CrowS-Pairs
- [x] XCopa
- [ ] BIG-Bench (Hailey)
- [x] XStoryCloze
- [x] XWinograd
- [x] PAWS-X
- [x] XNLI
- [x] MGSM
- [ ] SCROLLS
- [x] Babi
- [x] Belebele

# Novel Tasks
Tasks added in the revamped harness that were not previously available. Again, a strikethrough denotes checking performed *against the original task's implementation or published results introducing the task*.

# Task Wishlist

- [ ] TheoremQA
- [ ] Theorem Proving evaluations
- [ ] Chain of Thought
- [ ] Self-consistency ; Least-to-Most prompting, etc.
- [ ] Summarization Tasks
- [ ] Anthropic Model-Written Evals
