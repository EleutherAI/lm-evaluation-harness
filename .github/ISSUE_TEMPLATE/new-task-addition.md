---
name: New Task Addition
about: Template for proposing a new task to the benchmark
title: "[Task]"
labels: ''
assignees: ''

---

## Task Categorization
<!-- Please fill in the following information about your task -->

**Category**: <!-- Choose one: advanced_reasoning, basic_knowledge, research -->
**Subject**: <!-- e.g., Chemistry, Physics, Materials Science -->
**Domain**: <!-- Specific domain within the subject, e.g., Organic Chemistry, Quantum Mechanics -->
**Path**: <!-- Proposed path in repo structure, e.g., science/advanced_reasoning/chemistry/organic/ -->

## Task Description

**Task Name**: <!-- A short, descriptive name for your task -->

**Background**:
<!-- Provide context about why this task is important and what it aims to evaluate -->

**Main Goal**:
<!-- Clearly state what aspect of scientific knowledge or reasoning this task evaluates -->

**Expected Behavior**:
<!-- Describe what a successful response to this task looks like -->

## Dataset Information

**Source**:
<!-- Describe where the data comes from (e.g., published dataset, curated from papers) -->
<!-- Include relevant citations or links -->

**Dataset Size**:
<!-- Provide approximate numbers for:
- Training examples (if applicable)
- Validation examples
- Test examples
-->

**Data Format**:
<!-- Describe the format of your data (e.g., multiple choice, free response, structured output) -->

## Sample Implementation

**Input Example**:
```
<!-- Provide a representative example of task input -->
```

**Expected Output**:
```
<!-- Show what the correct output should look like -->
```

**Evaluation Metric**:
<!-- Describe how responses will be evaluated (e.g., exact match, RMSE, domain-specific metrics) -->

## Implementation Plan

**Required Files**:
- [ ] `task_name.yaml`
- [ ] `utils.py` (if needed)
- [ ] `README.md`

**Dependencies**:
<!-- List any special dependencies or requirements -->

## Checklist

- [ ] This task requires genuine scientific understanding
- [ ] This task is not solvable by simple pattern matching
- [ ] The task evaluation metrics are well-defined
- [ ] Sample input/output pairs are provided
- [ ] Dataset source is properly cited
- [ ] Implementation path follows repository structure

## Additional Context
<!-- Add any other relevant information, clarifications, or screenshots -->

## Alternative Approaches
<!-- If applicable, describe any alternative ways you considered implementing this task -->

## Related Issues/PRs
<!-- Reference any related issues or pull requests -->

---
<!-- Before submitting:
1. Tag appropriate maintainers
2. Add relevant labels (e.g., 'chemistry', 'physics', 'materials')
3. Link to any relevant documentation or papers
-->
