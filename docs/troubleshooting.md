
# Installation issues

If you run into issues with an editable installation, try enabling the legacy behaviour of `setuptools` first: 

```bash
export SETUPTOOLS_ENABLE_FEATURES="legacy-editable"
pip install -e .
```

if you still run into issues, as is often recommended, it's best to upgrade to `pip` and `Python`, and related dependencies and try again before reporting the issue. 