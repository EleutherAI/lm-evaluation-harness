# Clinical Safety — Turkish Clinical Source Support

**Group:** `clinical_safety`
**Task:** `turkish_clinical_source_support`

## Description

This task evaluates whether large language models (LLMs) can correctly identify evidence-based clinical recommendations in Turkish, distinguishing safe, guideline-supported answers from unsafe or incorrect medical advice.

The task consists of 10 multiple-choice questions covering common clinical scenarios in internal medicine:
- Diabetes management during acute illness
- Anticoagulation and drug interactions
- COPD exacerbation treatment
- Contrast nephropathy prevention
- Febrile neutropenia management
- Hyperkalemia treatment
- Metformin safety and lactic acidosis risk
- Perioperative DOAC management
- Opioid-induced constipation prophylaxis
- ACE inhibitor monitoring

Each question presents a clinical scenario in Turkish with four answer choices. Only one answer aligns with current evidence-based clinical guidelines. The other options represent common errors, outdated practices, or frankly unsafe recommendations — allowing evaluation of whether an LLM can reject unsafe medical advice.

## Boundary Statement

⚠️ **Important limitations:**

1. **No patient data:** This benchmark contains only synthetic clinical knowledge questions. It does NOT include any real patient data, protected health information (PHI), or identifiable clinical records.

2. **No clinical validation:** The questions and answers were written by a physician but have NOT been validated through formal clinical studies or inter-rater reliability testing. This is a research benchmark, not a clinical decision support tool.

3. **No ranking of models for clinical use:** Performance on this task does NOT certify any model as safe for clinical deployment. It is one narrow signal among many needed to evaluate medical AI safety. Do not use these scores to rank LLMs for real-world medical use.

4. **Guidelines evolve:** Clinical guidelines change over time. The answers reflect evidence as of 2025 and may become outdated.

5. **Turkish language only:** This task is designed for Turkish-language clinical evaluation and is not a general medical knowledge benchmark.

## Dataset Format

- **Format:** JSONL (one question per line)
- **Language:** Turkish
- **Samples:** 10 questions
- **Choices per question:** 4 (A, B, C, D)
- **Output type:** Multiple choice (loglikelihood comparison)

## Usage

```bash
lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks clinical_safety \
    --device cuda:0
```

Or run the single task directly:

```bash
lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks turkish_clinical_source_support \
    --device cuda:0
```

## Task Validity Checklist

- [ ] Is the task an existing benchmark in the literature?
  - [ ] Have you referenced the original paper that introduced the task?
  - [x] This is a novel task, not a reproduction of existing work.
- [x] Is the "Main" variant of this task clearly denoted?
  - [x] Yes — `turkish_clinical_source_support` is the only variant.
- [x] Have you provided a short sentence in a README on what this variant evaluates?
  - [x] Evaluates identification of evidence-based Turkish clinical recommendations.
- [x] Have you noted which, if any, published evaluation setups are matched by this variant?
  - [x] This is a novel evaluation setup with no direct published precedent.

## Author

Dr. Goktug Ozkan — Internal Medicine Specialist, Turkey

## License

This task is contributed under the same license as the lm-evaluation-harness repository (MIT).
