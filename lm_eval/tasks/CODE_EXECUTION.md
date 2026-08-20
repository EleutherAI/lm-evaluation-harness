# External code-execution protocol

HumanEval, MBPP, EvalPlus, and CRUXEval functional-correctness metrics use
`LM_EVAL_PYTHON_EXECUTOR`. MultiPL-E uses the equivalent language-aware
`LM_EVAL_MULTIPLE_EXECUTOR` protocol documented in its task README. There is no
in-process fallback.

The Python executor receives one JSON object per input line:

```json
{"schema_version": 1, "id": 0, "language": "python", "program": "..."}
```

It returns one line per program, in order, with the same `id` and either
`{"passed": true}` or the official-style `{"status": "OK", "exit_code": 0}`.
`LM_EVAL_PYTHON_EXECUTOR_TIMEOUT` controls the batch client timeout and defaults
to 600 seconds.

The command must launch programs outside the model worker with no network or
service-account token, a non-root identity, read-only root filesystem, private
temporary workspace, default seccomp/AppArmor, no Linux capabilities, and
strict CPU, process, memory, output, and wall-time limits. The lm-eval
`--confirm_run_unsafe_code` flag remains a separate required acknowledgement;
neither gate is a substitute for sandbox isolation.
