# Noor – MMLU-var Filtered Benchmark

**Authors:** Mohammed Dahbani & Anas Ezzakri (IMT Atlantique)

## Description

This benchmark is a **filtered and improved version of MMLU/MMLU-var**, designed to provide a **stable, monotonic, and informative evaluation signal during the early stages of LLM training**.

Standard MMLU often becomes noisy or non-discriminative for models that have seen only limited training. Our benchmark keeps only the questions that reliably reflect **true learning progress**.

## Methodology

### 1. Scientific Compliance Filtering
- Automatically retain all *hard-science* subjects (Math, Physics, Chemistry, etc.).
- For more variable subjects (Biology, Medicine, Humanities), apply an **LLM-as-a-Judge** process:
  - Each question is evaluated **5 times**.
  - A question is retained only if it receives **5/5 “Accept”** decisions.

This ensures clarity, consistency, and the removal of ambiguous items.

### 2. Signal Quality Filtering
For every question, we compute its **Confidence Margin** across all training checkpoints and fit a **linear regression**:

- Only questions with a **positive slope** are retained.
- This ensures that each item produces a **smooth and monotonic learning trend**.

The combination of both filters produces a benchmark that is cleaner, more stable, and much more sensitive to early-stage learning dynamics.

## Task Structure

The main group:
- `noor`

This includes all subjects that pass both filtering stages.

Each subject is also available as an independent task:
- Example: `noor_abstract_algebra`, `noor_college_physics`, `noor_machine_learning`, etc.

## Purpose

To provide a **reliable and low-noise evaluation signal** for early-stage LLMs, where traditional benchmarks usually fail to capture meaningful progress.
