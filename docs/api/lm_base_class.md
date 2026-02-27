# LM Base Class

Abstract base class for language models. Subclass this to add a new model backend to the evaluation harness.

[:material-github: Source](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py){ .md-button }

::: lm_eval.api.model.LM

---

`TemplateLM` provides common tokenization and chat template logic. Most built-in backends extend this rather than `LM` directly.

::: lm_eval.api.model.TemplateLM

---

`CachingLM` wraps any `LM` instance to add response caching.

::: lm_eval.api.model.CachingLM
