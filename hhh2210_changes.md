# hhh2210's Changes to lm-evaluation-harness

This document contains all changes made by hhh2210 (hzy2210@gmail.com) to the lm-evaluation-harness project.

## Summary

hhh2210 made 2 commits that focused on improving the handling of raw generations in model classes and adding support for Chain-of-Thought (COT) trace response handling.

## Commits

### 1. feat: COT trace response handling in evaluator and model classes
**Commit Hash:** 3c055832da324a0061e7ed9946b2ae317eaefd37  
**Date:** Sun Aug 3 12:36:20 2025 +0800

**Description:**
- Added support for storing raw generations in HFLM and VLLM models.
- Updated the evaluator to log warnings when the length of raw generations does not match processed responses.
- Modified response collection to include raw responses when available.

This improves the evaluation process by allowing access to the original generated outputs alongside processed responses.

**Files Modified:**
- `lm_eval/evaluator.py` (19 additions, 2 deletions)
- `lm_eval/models/huggingface.py` (12 additions, 4 deletions)  
- `lm_eval/models/vllm_causallms.py` (13 additions, 5 deletions)

**Key Changes:**

#### lm_eval/evaluator.py
- Added logic to extract raw responses when available from models
- Added warning logging when raw generations length doesn't match processed responses
- Modified response collection to use raw responses when available instead of processed ones
- Enhanced the data structure to include raw responses alongside filtered responses

#### lm_eval/models/huggingface.py
- Added `raw_generations` attribute initialization
- Modified generation process to store raw decoded text before post-processing
- Separated raw text storage from processed text for thinking tokens handling
- Updated cache hook to use raw generated text

#### lm_eval/models/vllm_causallms.py
- Added `raw_generations` attribute initialization
- Modified generation process to store raw generated text before post-processing
- Separated raw text caching from processed text handling
- Enhanced generation pipeline to maintain both raw and processed outputs

### 2. Refactor raw generation handling in VLLM model
**Commit Hash:** 12faad7ebb123fe27a2a197171cd821b75290fd6  
**Date:** Sun Aug 3 13:37:50 2025 +0800

**Description:**
- Introduced a buffer for raw generations during the generation process to improve performance and maintainability.
- Updated the final assignment of raw generations to ensure they are reordered correctly after processing.

This change enhances the clarity of the code and optimizes the handling of generated outputs. Otherwise, the log-samples would be mis-matched.

**Files Modified:**
- `lm_eval/models/vllm_causallms.py` (4 additions, 2 deletions)

**Key Changes:**

#### lm_eval/models/vllm_causallms.py
- Replaced direct assignment to `self.raw_generations` with a local buffer `raw_generations_buffer`
- Added proper reordering of raw generations at the end of the generation process
- Ensured raw generations maintain the same order as processed results
- Fixed potential mis-matching between log samples and their corresponding raw generations

### 3. fix: De-duplicate logged samples and sanitize stop sequences for CoT
**Date:** Sat Aug 9 2025 +0800

**Description:**
- Avoid duplicate per-document sample logs when tasks define multiple output filters (e.g., `strict-match` and `flexible-extract`). Metrics are still computed for all filters, but `logged_samples` are written only once per `doc_id`, preferring `flexible-extract` when available. This prevents downstream splitting from double-counting samples.
- Sanitize generation stop sequences to prevent premature truncation in Chain-of-Thought outputs. Specifically, the literal string `"Question:"` is removed from the stop list before decoding, since many CoT traces may echo "Question:" and should not be halted early.

**Files Modified:**
- `lm_eval/evaluator.py` (log-samples de-duplication; prefer `flexible-extract` for logging only)
- `lm_eval/models/vllm_causallms.py` (stop sequence sanitization: remove `"Question:"`)
- `lm_eval/models/huggingface.py` (stop sequence sanitization: remove `"Question:"`)

**Key Changes:**

#### lm_eval/evaluator.py
- Group instances by `doc_id` as before, but when `log_samples=True`, only write a log entry for one preferred filter (defaults to `flexible-extract` if present, otherwise the first available filter). Metrics aggregation across all filters is unchanged.

#### lm_eval/models/vllm_causallms.py and lm_eval/models/huggingface.py
- After resolving `until` via `handle_stop_sequences(...)`, filter out entries equal to `"Question:"` (case-insensitive, trimmed) to avoid early termination of CoT generations.

**Rationale:**
- De-duplication eliminates inflated sample counts (e.g., 2× test size when both filters were logged), stabilizing downstream splitting/analysis.
- Stop-seq sanitization prevents CoT outputs from being truncated when they naturally echo the word "Question:", improving answer completeness and restoring expected accuracy.

## Technical Impact

These changes collectively improve the evaluation framework by:

1. **Enhanced Traceability**: Raw generations are now preserved alongside processed outputs, enabling better debugging and analysis of model behavior.

2. **Improved Data Integrity**: The refactoring ensures that raw generations are properly ordered and matched with their corresponding processed outputs, preventing data misalignment.

3. **Better Debugging Support**: Evaluators can now access both raw and processed responses, making it easier to understand how post-processing affects model outputs.

4. **Consistent Implementation**: Both HFLM and VLLM models now handle raw generations in a consistent manner.

## Code Quality Improvements

- Proper separation of concerns between raw text generation and post-processing
- Better error handling with warning messages for mismatched response lengths  
- Cleaner code structure with dedicated buffers for temporary data
- Enhanced maintainability through consistent patterns across model implementations