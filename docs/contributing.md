# Contributing to LM Evaluation Harness

Welcome! We appreciate contributions and feedback.

## Important Resources

- [Documentation](https://lm-evaluation-harness.readthedocs.io/)
- [GitHub Milestones](https://github.com/EleutherAI/lm-evaluation-harness/milestones) for near-term release tracking
- [Project Board](https://github.com/orgs/EleutherAI/projects/25) for work items and roadmap
- [#lm-thunderdome](https://discord.gg/eleutherai) on EleutherAI Discord for discussion and support

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting via [pre-commit](https://pre-commit.com/).

```bash
pip install -e ".[dev]"
pre-commit install
```

This ensures linters and checks run on every commit.

## Testing

We use [pytest](https://docs.pytest.org/en/latest/) for unit tests:

```bash
python -m pytest --showlocals -s -vv -n=auto --ignore=tests/models/test_openvino.py
```

## Verbose logging

Enable debug logging with:

```bash
export LMEVAL_LOG_LEVEL="debug"
```

## Contributor License Agreement

First-time contributors must agree to a Contributor License Agreement (CLA). @CLAassistant will comment on your first PR with instructions.

## Contribution Best Practices

**For Pull Requests:**

- Descriptive title and brief description of scope and intent
- New features should include appropriate documentation
- Aim for code maintainability and minimize code copying
- Task PRs: share test results using a publicly-available model and compare to published results

**For Feature Requests:**

- Describe the feature, its motivation, and a use case
- Explain how it differs from current functionality

**For Bug Reports:**

- Provide a reproducible example (the exact command that triggers the error)
- Include the full error traceback
- Note your codebase version and relevant environment details

**For Requesting New Tasks:**

- 1-2 sentence description of what the task evaluates
- Links to: the paper, the dataset, results on open-source models, and any reference implementation

## Documentation Style

We use [mkdocstrings](https://mkdocstrings.github.io/) with Google-style docstrings. For details on formatting docstrings for the auto-generated API reference, see the [Docstring Guide](docstring_guide.md).

Key conventions:

- Google-style docstrings (not Sphinx/RST)
- Markdown cross-references: `[text][fully.qualified.path]`
- Inline code: double backticks `` `code` ``
- Code examples: `Example:` (singular) + fenced ` ```python ` blocks

## How Can I Get Involved?

Start with a [good first issue](https://github.com/EleutherAI/lm-evaluation-harness/issues?q=is%3Aopen+label%3A%22good+first+issue%22+label%3A%22help+wanted%22) or check the [project board](https://github.com/orgs/EleutherAI/projects/25/views/8).

Ways to contribute:

- **New evaluation tasks** — implement and verify benchmarks
- **Documentation** — improve guides, note gaps, fix errors
- **Testing** — add tests, improve CI/CD workflows
- **Model integrations** — add support for new inference libraries
- **New features** — open an issue to discuss before implementing

Questions? Join us on [Discord](https://discord.gg/eleutherai).
