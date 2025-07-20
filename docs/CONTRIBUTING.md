# Contributing to LM Evaluation Harness

Welcome and thank you for your interest in the LM Evaluation Harness! We welcome contributions and feedback and appreciate your time spent with our library, and hope you find it useful!

## Important Resources

There are several places information about LM Evaluation Harness is located:

- Our [documentation pages](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs)
- We occasionally use [GitHub Milestones](https://github.com/EleutherAI/lm-evaluation-harness/milestones) to track progress toward specific near-term version releases.
- We maintain a [Project Board](https://github.com/orgs/EleutherAI/projects/25) for tracking current work items and PRs, and for future roadmap items or feature requests.
- Further discussion and support conversations are located in the #lm-thunderdome channel of the [EleutherAI discord](https://discord.gg/eleutherai).

## Code Style

LM Evaluation Harness uses [ruff](https://github.com/astral-sh/ruff) for linting via [pre-commit](https://pre-commit.com/).

You can install linters and dev tools via

```pip install lm_eval[dev]``` or ```pip install -e ".[dev]"```

Then, run

```pre-commit install```

in order to ensure linters and other checks will be run upon committing.

## Testing

We use [pytest](https://docs.pytest.org/en/latest/) for running unit tests. All library unit tests can be run via:

```bash
python -m pytest --showlocals -s -vv -n=auto --ignore=tests/models/test_openvino.py
```

## Contributor License Agreement

We ask that new contributors agree to a Contributor License Agreement affirming that EleutherAI has the rights to use your contribution to our library.
First-time pull requests will have a reply added by @CLAassistant containing instructions for how to confirm this, and we require it before merging your PR.

## Contribution Best Practices

We recommend a few best practices to make your contributions or reported errors easier to assist with.

**For Pull Requests:**

- PRs should be titled descriptively, and be opened with a brief description of the scope and intent of the new contribution.
- New features should have appropriate documentation added alongside them.
- Aim for code maintainability, and minimize code copying.
- If opening a task, try to share test results on the task using a publicly-available model, and if any public results are available on the task, compare to them.

**For Feature Requests:**

- Provide a short paragraph's worth of description. What is the feature you are requesting? What is its motivation, and an example use case of it? How does this differ from what is currently supported?

**For Bug Reports**:

- Provide a short description of the bug.
- Provide a *reproducible example*--what is the command you run with our library that results in this error? Have you tried any other steps to resolve it?
- Provide a *full error traceback* of the error that occurs, if applicable. A one-line error message or small screenshot snippet is unhelpful without the surrounding context.
- Note what version of the codebase you are using, and any specifics of your environment and setup that may be relevant.

**For Requesting New Tasks**:

- Provide a 1-2 sentence description of what the task is and what it evaluates.
- Provide a link to the paper introducing the task.
- Provide a link to where the dataset can be found.
- Provide a link to a paper containing results on an open-source model on the task, for use in comparisons and implementation validation.
- If applicable, link to any codebase that has implemented the task (especially the original publication's codebase, if existent).

## How Can I Get Involved?

To quickly get started, we maintain a list of good first issues, which can be found [on our project board](https://github.com/orgs/EleutherAI/projects/25/views/8) or by [filtering GH Issues](https://github.com/EleutherAI/lm-evaluation-harness/issues?q=is%3Aopen+label%3A%22good+first+issue%22+label%3A%22help+wanted%22). These are typically smaller code changes or self-contained features which can be added without extensive familiarity with library internals, and we recommend new contributors consider taking a stab at one of these first if they are feeling uncertain where to begin.

There are a number of distinct ways to contribute to LM Evaluation Harness, and all are extremely helpful! A sampling of ways to contribute include:

- **Implementing and verifying new evaluation tasks**: Is there a task you'd like to see LM Evaluation Harness support? Consider opening an issue requesting it, or helping add it! Verifying and cross-checking task implementations with their original versions is also a very valuable form of assistance in ensuring standardized evaluation.
- **Improving documentation** - Improvements to the documentation, or noting pain points / gaps in documentation, are helpful in order for us to improve the user experience of the library and clarity + coverage of documentation.
- **Testing and devops** - We are very grateful for any assistance in adding tests for the library that can be run for new PRs, and other devops workflows.
- **Adding new modeling / inference library integrations** - We hope to support a broad range of commonly-used inference libraries popular among the community, and welcome PRs for new integrations, so long as they are documented properly and maintainable.
- **Proposing or Contributing New Features** - We want LM Evaluation Harness to support a broad range of evaluation usecases. If you have a feature that is not currently supported but desired, feel free to open an issue describing the feature and, if applicable, how you intend to implement it. We would be happy to give feedback on the cleanest way to implement new functionalities and are happy to coordinate with interested contributors via GH discussions or via discord.

We hope that this has been helpful, and appreciate your interest in contributing! Further questions can be directed to [our Discord](discord.gg/eleutherai).
