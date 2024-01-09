# Contributing to LM Evaluation Harness

Welcome and thank you for your interest in the LM Evaluation Harness! We welcome contributions and feedback and appreciate your time spent with our library, and hope you find it useful!

## Important Resources

There are several places information about LM Evaluation Harness is located: 

- Our [documentation pages](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) 
- We occasionally use [GitHub Milestones](https://github.com/EleutherAI/lm-evaluation-harness/milestones) to track progress toward specific near-term version releases.
- We maintain a [Project Board](https://github.com/orgs/EleutherAI/projects/25) for tracking current work items and PRs, and for future roadmap items or feature requests. 
- Further discussion and support conversations are located in the #lm-thunderdome channel of the [EleutherAI discord](discord.gg/eleutherai).

## Code Style

LM Evaluation Harness uses [ruff](https://github.com/astral-sh/ruff) for linting via [pre-commit](https://pre-commit.com/). 

You can install linters and dev tools via 

```pip install lm_eval[dev]```

Then, run 

```pre-commit install```

in order to ensure linters and other checks will be run upon committing.

## Testing

We use [pytest](https://docs.pytest.org/en/latest/) for running unit tests. All library unit tests can be run via:

```
python -m pytest --ignore=tests/tests_master --ignore=tests/extra
```

## Contributor License Agreement

We ask that new contributors agree to a Contributor License Agreement affirming that EleutherAI has the rights to use your contribution to our library. 
First-time pull requests will have a reply added by @CLAassistant containing instructions for how to confirm this, and we require it before merging your PR. 

## Guide to contributing


## Different Contribution Types

- Implementing and verifying new tasks
- Improving documentation + testing + automation / devops
- New features--coordinate with us
- Proposing new features / additions
- new internal/external model library integrations


## How Can I Get Involved?

There are a number of ways to 

- Good First Issues board / adding new tasks
