# LiveCodeBench Logging Configuration

## Overview

The LiveCodeBench evaluation logging has been improved to reduce redundancy and make output more readable.

## Key Improvements

1. **Removed Duplicate Logging**: Eliminated duplicate stderr output that was causing identical lines to appear twice
2. **Summary Mode by Default**: Only shows test pass/fail status without full input/output details by default
3. **Failure Details**: Still shows full input/output details when tests fail, for debugging
4. **Cleaner Progress Tracking**: Simplified progress messages without duplication

## Default Behavior

By default, the logging is now in "summary mode":
- ✅ Shows pass/fail status for each test
- ✅ Shows full details only for failed tests 
- ✅ Shows problem-level results and progress
- ❌ Does NOT show full input/output for passing tests

## Enabling Verbose Mode

If you need to see full input/output details for all tests (like the original behavior), you can enable verbose mode:

```python
from lm_eval.tasks.livecodebench.utils import configure_livecodebench_logging

# Enable verbose logging (shows all test details)
configure_livecodebench_logging(verbose=True)

# Disable verbose logging (summary mode, default)
configure_livecodebench_logging(verbose=False)
```

## Example Output Comparison

### Summary Mode (Default)
```
🧪 Testing stdio-based problem with 13 test cases:
Test 1: ✅ PASS
Test 2: ✅ PASS
Test 3: ❌ FAIL
Input: 8\n6 3\nWBWWWB\n...
Generated Output: 0\n0\n...
Ground Truth: 2\n1\n...
Error: Line 0: '0' != '2'

📊 Problem Result: 0/13 tests passed - FAILED
```

### Verbose Mode
```
🧪 Testing stdio-based problem with 13 test cases:
Test 1: ✅ PASS
Input: 8\n6 3\nWBWWWB\n...
Generated Output: 2\n1\n...
Ground Truth: 2\n1\n...
Test 2: ✅ PASS
Input: ...
Generated Output: ...
Ground Truth: ...
[... continues for all tests ...]
```

The summary mode reduces log file size significantly while maintaining all necessary debugging information for failed cases. 