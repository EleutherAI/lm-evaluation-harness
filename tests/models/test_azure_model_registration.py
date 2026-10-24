import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("first_model", [None, "dummy", "local-chat-completions"])
def test_azure_chat_resolves_without_prior_backend_import(first_model):
    # Other tests import openai_completions during collection, which can mask a
    # missing lazy entry. Resolve in a fresh interpreter, as the CLI does.
    code = textwrap.dedent(
        f"""
        import sys
        from lm_eval.api.registry import get_model

        assert 'lm_eval.models.openai_completions' not in sys.modules
        first_model = {first_model!r}
        if first_model is not None:
            get_model(first_model)
        model = get_model('azure-openai-chat-completions')

        from lm_eval.models.openai_completions import AzureOpenaiChatCompletionsLM
        assert model is AzureOpenaiChatCompletionsLM
        assert get_model('azure-openai-chat-completions') is model
        """
    )
    result = subprocess.run(  # noqa: S603 - fixed test code in the current interpreter
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
