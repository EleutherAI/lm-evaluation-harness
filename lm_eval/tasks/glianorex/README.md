# Glianorex

The goal of this benchmark is to isolate the test answering capabilities from the content knowledge.

### Paper

Title: Multiple Choice Questions and Large Languages Models: A Case Study with Fictional Medical Data

Abstract: https://arxiv.org/abs/2406.02394

To test the relevance of MCQs to assess LLM performance without prior data exposure, we created a fictional medical benchmark and knowledge base on a non-existent gland, the Glianorex. Using GPT-4 we generated a comprehensive textbook on the Glianorex in both English and French, and created multiple-choice questions in both English and French.

### Tasks

All tasks are multiple choice questions with 4 options, only one correct option.

- `glianorex`: Evaluates all tasks listed below.

- `glianorex_en`: Evaluates the accuracy on 264 questions in English.
- `glianorex_fr`: Evaluates the accuracy on 264 questions in French.
