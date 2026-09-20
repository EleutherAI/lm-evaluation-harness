# DETAILS.md

🔍 **Powered by [Detailer](https://detailer.ginylil.com)** - Intelligent agent-ready documentation



---

## 1. Project Overview

### Purpose & Domain
This project is an extensive **language model evaluation framework** named `lm_eval`, designed to benchmark and analyze large language models (LLMs) across a wide variety of **natural language processing (NLP) tasks**, with a strong focus on **multilingual and African language datasets**.

### Problem Solved
- Provides a **standardized, modular, and extensible evaluation harness** for LLMs.
- Enables **systematic benchmarking** across diverse tasks such as translation, sentiment analysis, named entity recognition (NER), natural language inference (NLI), question answering, and more.
- Supports **multilingual and low-resource languages**, addressing gaps in existing evaluation frameworks.
- Facilitates **prompt engineering** and **few-shot/zero-shot evaluation** with configurable prompt templates.
- Offers **integration with multiple model backends** (e.g., HuggingFace, OpenAI, Anthropic, IBM Watsonx).
- Supports **robust metric computation** and **result aggregation** for comprehensive model assessment.

### Target Users and Use Cases
- **Researchers and practitioners** developing or benchmarking LLMs.
- **Multilingual NLP developers** focusing on African languages and other low-resource languages.
- **Data scientists and ML engineers** needing flexible evaluation pipelines.
- **Prompt engineers** experimenting with prompt templates and few-shot learning.
- **Organizations** aiming to evaluate model performance across diverse tasks and languages.

### Value Proposition
- **Comprehensive task coverage** with hundreds of tasks and datasets.
- **Modular architecture** enabling easy addition of new tasks, models, and prompt templates.
- **Support for multiple model APIs and local models**, including advanced tokenization and batching.
- **Automated configuration generation** for datasets and prompts.
- **Rich metric implementations** with bootstrapping and statistical analysis.
- **Operational tooling** including CI/CD workflows, monitoring, and logging integrations.

---

## 2. Architecture and Structure

### Complete Repository Structure (Summary)

```
.
├── .github/workflows/                  # CI/CD pipelines (testing, publishing)
├── docs/                              # Documentation (API guides, task guides, footguns)
├── examples/                          # Example notebooks and scripts
├── lm_eval/                          # Core evaluation framework (12990 files)
│   ├── api/                         # Core API abstractions (models, tasks, metrics, filters)
│   ├── caching/                     # Caching utilities
│   ├── decontamination/             # Dataset decontamination utilities
│   ├── filters/                     # Post-processing filters (regex, voting, etc.)
│   ├── loggers/                     # Logging and result tracking (WandB, HF Hub)
│   ├── models/                      # Model backends and wrappers (OpenAI, Anthropic, GGUF, etc.)
│   ├── prompts/                     # Prompt templates and utilities
│   ├── tasks/                       # Task definitions and datasets (~12929 files)
│   │   ├── aclue/                   # ACLUE benchmark tasks
│   │   ├── acpbench/                # ACPBench tasks (BoolQ, MCQ, etc.)
│   │   ├── afrimgsm/                # African multilingual tasks (direct, cot, translate)
│   │   ├── afrimmlu/                # African MMLU tasks
│   │   ├── afrobench/               # AfroBench benchmark suite (AfriSenti, MasakhaNER, etc.)
│   │   │   ├── adr/                 # Automatic Diacritics Restoration tasks
│   │   │   ├── afriqa/              # AfriQA question answering tasks
│   │   │   ├── afrisenti/           # AfriSenti sentiment analysis tasks
│   │   │   ├── belebele/            # Belebele multilingual tasks
│   │   │   ├── mafand/              # Mafand translation tasks
│   │   │   ├── masakhaner/          # MasakhaNER NER tasks
│   │   │   ├── masakhapos/          # MasakhaPOS POS tagging tasks
│   │   │   ├── masakhanews/         # MasakhaNEWS classification tasks
│   │   │   ├── nollysenti/          # Nollywood sentiment analysis tasks
│   │   │   ├── ntrex/               # NTREX translation tasks
│   │   │   ├── openai_mmlu/         # OpenAI MMLU tasks
│   │   │   ├── salt/                # SALT translation tasks
│   │   │   ├── sib/                 # SIB classification tasks
│   │   │   └── ...                  # Other task directories
│   ├── __init__.py                  # Package initialization
│   ├── evaluator.py                 # Evaluation orchestration
│   ├── evaluator_utils.py           # Evaluation utilities
│   └── utils.py                    # General utilities (hashing, YAML loading, etc.)
├── scripts/                        # Utility scripts (data cleaning, benchmark building)
├── templates/                     # Template files for YAML generation
├── tests/                        # Extensive test suite (~742 files)
├── .gitignore
├── .pre-commit-config.yaml
├── CITATION.bib
├── LICENSE.md
├── MANIFEST.in
├── README.md
├── pyproject.toml
├── requirements.txt
└── setup.py
```

### File Location Context
- **Task Definitions:** Located under `lm_eval/tasks/`, tasks are organized by benchmark and dataset, with subdirectories for prompt variants and languages.
- **Model Backends:** Under `lm_eval/models/`, supporting multiple APIs and local models.
- **Utilities:** `lm_eval/utils.py` and task-specific `utils.py` files provide helper functions.
- **Prompt Templates:** Stored under `lm_eval/prompts/` and within task directories.
- **Logging & Monitoring:** Under `lm_eval/loggers/`, integrating with WandB and Hugging Face Hub.
- **CI/CD:** GitHub workflows automate testing, publishing, and task detection.

---

## 3. Technical Implementation Details

### Core Components

- **API Layer (`lm_eval/api/`):**  
  Defines abstract base classes and interfaces for models (`LM`), tasks (`Task`), filters, metrics, and grouping. Uses decorators for registration (`@register_model`, `@register_task`).

- **Model Implementations (`lm_eval/models/`):**  
  Implements wrappers for various LLMs:
  - OpenAI API (`openai_completions.py`)
  - Anthropic API (`anthropic_llms.py`)
  - GGUF local models (`gguf.py`)
  - HuggingFace models (`huggingface.py`)
  - IBM Watsonx (`ibm_watsonx_ai.py`)
  - VLLM-based models (`vllm_causallms.py`, `vllm_vlms.py`)
  
  Supports tokenization backends (`tiktoken`, `huggingface`), batching, retries, and concurrency.

- **Task Definitions (`lm_eval/tasks/`):**  
  Each task directory contains:
  - YAML files defining datasets, prompt templates, and task metadata.
  - Utility scripts (`utils.py`, `gen_utils.py`) for generating YAML configs and prompt templates.
  - README files documenting datasets and tasks.
  - Prompt variants (`prompt_1`, `prompt_2`, etc.) supporting different prompt styles or few-shot settings.

- **Prompt Templates:**  
  Use Jinja2-like templating with placeholders (e.g., `{{question}}`, `{{sentence_eng_Latn}}`) for dynamic prompt generation.

- **Evaluation Metrics (`lm_eval/api/metrics.py`):**  
  Implements standard NLP metrics (accuracy, BLEU, F1, perplexity) and supports bootstrapping for confidence intervals.

- **Filters (`lm_eval/filters/`):**  
  Post-processing filters for model outputs, including regex extraction, majority voting, whitespace trimming, and decontamination.

- **Decontamination (`lm_eval/decontamination/`):**  
  Tools for dataset cleaning, contamination detection, and archival, including memory-mapped file readers and n-gram based contamination filtering.

- **Logging (`lm_eval/loggers/`):**  
  Integrates with WandB and Hugging Face Hub for experiment tracking, result storage, and metadata card generation.

- **Utilities (`lm_eval/utils.py`):**  
  Provides general-purpose functions for hashing, YAML loading, prompt templating, iterator slicing, and image processing.

### Development Patterns

- **Registration & Plugin System:**  
  Uses decorators to register models, tasks, filters, and metrics, enabling dynamic discovery and extensibility.

- **Template Method Pattern:**  
  Abstract base classes define workflows with overridable methods (e.g., `TemplateAPI` for API models).

- **Factory Pattern:**  
  Configuration generators (`gen_utils.py`) produce YAML files programmatically, supporting scalable dataset and prompt management.

- **Strategy Pattern:**  
  Different prompt templates and evaluation metrics are selected dynamically based on configuration.

- **Error Handling:**  
  Consistent use of try-except blocks, validation during YAML loading, and retries for API calls.

- **Testing:**  
  Extensive test suite covering models, tasks, filters, and utilities.

---

## 4. Development Patterns and Standards

- **Code Organization:**  
  - Modular directory structure separating API, models, tasks, filters, loggers, and utilities.
  - Task directories structured by benchmark and dataset, with prompt variants.

- **Coding Standards:**  
  - Use of type annotations and docstrings.
  - Consistent naming conventions for files, classes, and functions.
  - Use of standard Python libraries and third-party packages (`tqdm`, `requests`, `transformers`, `datasets`, `wandb`).

- **Configuration Management:**  
  - YAML files for declarative task and prompt configuration.
  - Use of includes and template inheritance to reduce duplication.

- **Testing:**  
  - Pytest-based tests covering core components.
  - Use of mock datasets and fixtures.

- **Documentation:**  
  - Markdown files for API guides, task descriptions, and usage instructions.
  - Inline comments and docstrings for clarity.

- **CI/CD:**  
  - GitHub Actions workflows for linting, testing, and publishing.

---

## 5. Integration and Dependencies

### External Dependencies

- **Model APIs:** OpenAI, Anthropic, IBM Watsonx, HuggingFace Transformers.
- **Tokenization:** `tiktoken`, `huggingface` tokenizers.
- **Data Processing:** `datasets`, `jsonlines`, `numpy`, `pandas`.
- **Evaluation Metrics:** `scikit-learn`, `sacrebleu`.
- **Logging & Tracking:** `wandb`, `huggingface_hub`.
- **Utilities:** `requests`, `aiohttp`, `tenacity` for retries.

### Internal Dependencies

- **Registration Decorators:** Centralized in `lm_eval/api/registry.py`.
- **Task and Model Interfaces:** Defined in `lm_eval/api/model.py` and `lm_eval/api/task.py`.
- **Prompt Templates and Filters:** Modular components in `lm_eval/prompts/` and `lm_eval/filters/`.
- **Dataset Configurations:** YAML files generated and managed via `gen_utils.py` and `utils.py` in task directories.

---

## 6. Usage and Operational Guidance

### Getting Started

- **Installation:**  
  Follow instructions in `README.md` and `requirements.txt` for dependencies.

- **Running Evaluations:**  
  Use CLI scripts or Python APIs to run evaluations on specified models and tasks.  
  Example:  
  ```bash
  python -m lm_eval --model <model_name> --tasks <task_names> --device <device> --batch_size <batch_size>
  ```

- **Adding New Tasks:**  
  - Create a new task directory under `lm_eval/tasks/` with YAML configs and prompt templates.  
  - Register the task via decorators in `lm_eval/api/registry.py`.  
  - Provide dataset processing and prompt generation logic as needed.

- **Adding New Models:**  
  - Implement a model wrapper in `lm_eval/models/`.  
  - Register the model with `@register_model`.  
  - Support required API methods (`loglikelihood`, `generate_until`, etc.).

- **Prompt Engineering:**  
  - Modify or add prompt templates in task directories using YAML files with Jinja2-like placeholders.  
  - Use `gen_utils.py` scripts for automated YAML generation.

- **Evaluation Metrics:**  
  - Use built-in metrics or add new ones in `lm_eval/api/metrics.py`.  
  - Register metrics with `@register_metric`.

### Monitoring and Logging

- **Weights & Biases Integration:**  
  - Configure WandB via environment variables.  
  - Use `lm_eval/loggers/wandb_logger.py` for logging metrics and artifacts.

- **Hugging Face Hub:**  
  - Use `lm_eval/loggers/evaluation_tracker.py` to push results and metadata.

### Configuration Management

- **YAML Files:**  
  - Central to task and prompt configuration.  
  - Use includes and inheritance to manage shared settings.

- **CLI Utilities:**  
  - Use `gen_utils.py` scripts in task directories to generate or update YAML configs.

### Testing

- Run tests via pytest:  
  ```bash
  pytest tests/
  ```

- Tests cover models, tasks, filters, and utilities.

---

## 7. Actionable Insights for AI Agents and Developers

- **Navigating the Codebase:**  
  - Start with `lm_eval/api/` for core interfaces and registration.  
  - Explore `lm_eval/models/` for model implementations and API wrappers.  
  - Review `lm_eval/tasks/` for task definitions, prompt templates, and dataset configurations.  
  - Use `gen_utils.py` scripts to understand YAML generation and prompt templating.

- **Extending the Framework:**  
  - To add a new model, implement the interface in `lm_eval/models/` and register it.  
  - To add a new task, create a directory under `lm_eval/tasks/` with YAML configs and prompt templates, register the task, and provide dataset processing code.  
  - To add new prompts, modify or add YAML files in the task's prompt directories.

- **Working with Prompts:**  
  - Prompts use Jinja2-like templating with placeholders (e.g., `{{question}}`, `{{sentence_eng_Latn}}`).  
  - Modify prompt YAML files to experiment with prompt engineering.

- **Managing Dependencies:**  
  - External dependencies are managed via `requirements.txt`.  
  - Use the provided GitHub Actions workflows for CI/CD and testing.

- **Operational Best Practices:**  
  - Use batching and concurrency features in model wrappers for efficient evaluation.  
  - Leverage caching mechanisms (`lm_eval/caching/`) to avoid redundant computations.  
  - Monitor experiments via WandB or Hugging Face Hub integrations.

- **Security and Data Integrity:**  
  - Use decontamination tools (`lm_eval/decontamination/`) to ensure dataset quality and avoid data leakage.

---

# End of DETAILS.md