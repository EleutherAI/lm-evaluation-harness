# GigaChat

The `gigachat` model backend uses the official [GigaChat Python SDK](https://github.com/ai-forever/gigachat). Install its optional dependencies with:

```sh
pip install "lm-eval[gigachat]"
```

Configure authentication using the SDK's environment variables. For example:

```sh
export GIGACHAT_CREDENTIALS="<authorization-key>"
export GIGACHAT_SCOPE="GIGACHAT_API_PERS"
```

Then run a generation-based task:

```sh
lm_eval \
    --model gigachat \
    --model_args model=GigaChat \
    --tasks gsm8k \
    --apply_chat_template
```

All SDK connection settings can also be supplied in `--model_args`, including `base_url`, `auth_url`, `access_token`, `verify_ssl_certs`, and `ca_bundle_file`. TLS certificate verification follows the SDK's secure default. See the [SDK configuration documentation](https://github.com/ai-forever/gigachat#configuration) for the complete list of `GIGACHAT_*` environment variables and authentication methods.

The GigaChat API does not expose prompt log probabilities, so `loglikelihood` and `loglikelihood_rolling` tasks are not supported.
