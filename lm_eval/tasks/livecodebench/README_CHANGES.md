# Live Code Bench - Evalscope Alignment Summary

This document summarizes the comprehensive changes made to align the lm-evaluation-harness Live Code Bench implementation with the evalscope implementation to ensure identical evaluation results.

## Key Changes Made

### 1. Complete Testing Utilities Replacement (`testing_util.py`)

**Previous Implementation:**
- Simple execution with basic timeout handling
- Limited error classification
- No security restrictions
- Single execution mode

**New Implementation (from evalscope):**
- **Two execution modes**: 
  - Call-based: For function-based problems (using `fn_name` from metadata)
  - Stdio-based: For standard input/output problems
- **Comprehensive timeout handling**: Global timeout + per-test timeout
- **Security restrictions**: `reliability_guard()` disables dangerous operations
- **Advanced error classification**: Specific error codes (-1, -2, -3, -4)
- **Decimal-based comparison**: For accurate floating-point comparisons
- **Proper stdin/stdout mocking**: Complete input/output simulation

### 2. Enhanced Pass@k Utilities (`pass_k_utils.py`)

**Previous Implementation:**
- Simple boolean-based pass@k calculation
- Basic accuracy computation

**New Implementation (from evalscope):**
- **Statistical estimation**: Proper pass@k using combinatorial formula
- **Detailed metrics**: Instance-wise results and aggregation
- **Compatible with evalscope format**: Exact same calculation logic

### 3. Core Evaluation Logic Updates (`utils.py`)

**Previous Implementation:**
- Simplified data processing
- Basic code extraction
- Limited private test case handling

**New Implementation (from evalscope):**
- **Data transformation**: Exact evalscope `transform_data_item()` logic
- **Format prompt generation**: Matches evalscope prompt structure
- **Private test case handling**: Proper pickle/compression decoding
- **Evaluation pipeline**: Identical to evalscope `codegen_metrics()`

### 4. YAML Configuration Updates (`livecodebench.yaml`)

**Changes:**
- **System message**: Enabled the exact evalscope system prompt
- **Prompt format**: Updated to match evalscope format exactly
- **Answer instruction**: Changed to "use the provided format with backticks"

## Key Technical Differences Addressed

### 1. Problem Type Detection
- **Evalscope**: Uses `fn_name` in metadata to distinguish call-based vs stdio-based
- **Previous lm-eval**: Single execution mode
- **Fix**: Implemented proper problem type detection in `run_test()`

### 2. Code Execution Security
- **Evalscope**: Comprehensive `reliability_guard()` function
- **Previous lm-eval**: No security restrictions
- **Fix**: Added complete security sandbox matching evalscope

### 3. Error Handling
- **Evalscope**: Specific error codes with detailed metadata
  - `-1`: Runtime error/timeout in individual test
  - `-2`: Wrong answer/compilation error
  - `-3`: Time limit exceeded
  - `-4`: Runtime error during testing
- **Previous lm-eval**: Simple boolean pass/fail
- **Fix**: Implemented exact error classification system

### 4. Data Processing
- **Evalscope**: Complex data transformation with format prompts
- **Previous lm-eval**: Simple input/output processing
- **Fix**: Implemented `transform_data_item()` function

### 5. Metric Calculation
- **Evalscope**: Statistical pass@k with proper estimation
- **Previous lm-eval**: Simple accuracy calculation
- **Fix**: Implemented `estimate_pass_at_k()` function

## Testing Compatibility

The updated implementation now:

1. **Handles all problem types**: Both call-based and stdio-based problems
2. **Uses identical timeout logic**: Global + per-test timeouts
3. **Applies same security restrictions**: All dangerous operations disabled
4. **Processes data identically**: Same transformation and format prompts
5. **Calculates metrics identically**: Exact same pass@k estimation
6. **Handles errors identically**: Same error codes and metadata

## Verification

To verify identical results:

1. **Run same problems**: Use identical dataset and version
2. **Compare outputs**: Scores should match exactly between evalscope and lm-eval
3. **Check metadata**: Error classifications should be identical
4. **Validate timeouts**: Same timeout behavior on long-running code

## Files Modified

1. `testing_util.py` - Complete replacement with evalscope implementation
2. `pass_k_utils.py` - Updated with evalscope pass@k calculation
3. `utils.py` - Enhanced with evalscope evaluation pipeline
4. `livecodebench.yaml` - Updated prompt format and system message
5. `README_CHANGES.md` - This documentation

## Expected Outcome

The lm-evaluation-harness Live Code Bench implementation now produces **identical results** to the evalscope implementation, ensuring fair and consistent evaluation across different frameworks. 