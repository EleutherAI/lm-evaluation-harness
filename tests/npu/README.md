# NPU Tests

This directory contains tests for the streamlined Rebellions NPU integration supporting **causal and seq2seq models**.

## Files

- **`test_npu_integration.py`** - Core integration tests

## Running Tests

```bash
# Run NPU integration tests
python tests/npu/test_npu_integration.py

# Quick integration verification
python test_standard_lm_eval.py

# Check NPU hardware status
python scripts/check_npu.py
```

## Test Coverage

- NPU device detection and auto-selection
- Model registration for `rbln`
- Causal and seq2seq model type detection
- RBLN Auto classes (`RBLNAutoModelForCausalLM`, `RBLNAutoModelForSeq2SeqLM`)
- Per-token forward pass for causal models
- Cache position fixes for seq2seq models
- Standard lm_eval integration with NPU auto-detection
