#!/usr/bin/env python3
"""Generate all YAML configuration files for LongBench v2 tasks."""

import os
import yaml

# Define all LongBench v2 tasks with their configurations
TASKS = {
    # Single-Document QA
    "narrativeqa": {
        "dataset_name": "narrativeqa",
        "prompt": "Read the following narrative and answer the question based on the information provided. Give a concise answer.\n\nNarrative:\n{{context}}\n\nQuestion: {{question}}\nAnswer:",
        "target": "{{answers[0] if answers else answer}}",
        "metric": "qa_f1",
        "max_gen_toks": 100,
    },
    "qasper": {
        "dataset_name": "qasper",
        "prompt": "Read the following scientific paper excerpt and answer the question.\n\nPaper:\n{{context}}\n\nQuestion: {{question}}\nAnswer:",
        "target": "{{answer}}",
        "metric": "qa_f1",
        "max_gen_toks": 100,
    },
    "multifieldqa": {
        "dataset_name": "multifieldqa_en",
        "prompt": "Answer the question based on the given context.\n\nContext:\n{{context}}\n\nQuestion: {{question}}\nAnswer:",
        "target": "{{answer}}",
        "metric": "qa_f1",
        "max_gen_toks": 50,
    },
    
    # Multi-Document QA
    "hotpotqa": {
        "dataset_name": "hotpotqa",
        "prompt": "Answer the following question by reasoning across the provided documents. Only give the answer without explanation.\n\nDocuments:\n{{context}}\n\nQuestion: {{question}}\nAnswer:",
        "target": "{{answer}}",
        "metric": "qa_f1",
        "max_gen_toks": 50,
    },
    "2wikimqa": {
        "dataset_name": "2wikimqa",
        "prompt": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{{context}}\n\nQuestion: {{input}}\nAnswer:",
        "target": "{{answers}}",
        "metric": "qa_f1",
        "max_gen_toks": 32,
    },
    "musique": {
        "dataset_name": "musique",
        "prompt": "Answer the question based on the given documents.\n\nDocuments:\n{{context}}\n\nQuestion: {{question}}\nAnswer:",
        "target": "{{answer}}",
        "metric": "qa_f1",
        "max_gen_toks": 50,
    },
    
    # Summarization
    "gov_report": {
        "dataset_name": "gov_report",
        "prompt": "Summarize the following government report.\n\nReport:\n{{context}}\n\nSummary:",
        "target": "{{summary}}",
        "metric": "rouge",
        "max_gen_toks": 200,
    },
    "multi_news": {
        "dataset_name": "multi_news",
        "prompt": "Summarize the following news articles.\n\nArticles:\n{{context}}\n\nSummary:",
        "target": "{{summary}}",
        "metric": "rouge",
        "max_gen_toks": 200,
    },
    "book_sum": {
        "dataset_name": "book_summarization",
        "prompt": "Summarize the following book excerpt.\n\nText:\n{{context}}\n\nSummary:",
        "target": "{{summary}}",
        "metric": "rouge",
        "max_gen_toks": 300,
    },
    
    # Few-shot Learning
    "trec": {
        "dataset_name": "trec",
        "prompt": "Classify the following question into one of these categories: ABBR, ENTY, DESC, HUM, LOC, NUM.\n\nExamples:\n{{context}}\n\nQuestion: {{question}}\nCategory:",
        "target": "{{answer}}",
        "metric": "classification",
        "max_gen_toks": 10,
    },
    "triviaqa": {
        "dataset_name": "triviaqa",
        "prompt": "Answer the trivia question based on the given context.\n\nContext:\n{{context}}\n\nQuestion: {{question}}\nAnswer:",
        "target": "{{answer}}",
        "metric": "qa_f1",
        "max_gen_toks": 50,
    },
    "samsum": {
        "dataset_name": "samsum",
        "prompt": "Summarize the following dialogue.\n\nDialogue:\n{{context}}\n\nSummary:",
        "target": "{{summary}}",
        "metric": "rouge",
        "max_gen_toks": 100,
    },
    
    # Synthetic Tasks
    "passage_retrieval": {
        "dataset_name": "passage_retrieval_en",
        "prompt": "Find the passage that answers the question.\n\nPassages:\n{{context}}\n\nQuestion: {{question}}\nPassage number:",
        "target": "{{answer}}",
        "metric": "retrieval",
        "max_gen_toks": 10,
    },
    "passage_count": {
        "dataset_name": "passage_count",
        "prompt": "Count the number of passages in the following text.\n\nText:\n{{context}}\n\nNumber of passages:",
        "target": "{{answer}}",
        "metric": "count",
        "max_gen_toks": 10,
    },
    "kv_retrieval": {
        "dataset_name": "kv_retrieval",
        "prompt": "Find the value for the given key in the following key-value pairs.\n\nKey-Value Pairs:\n{{context}}\n\nKey: {{key}}\nValue:",
        "target": "{{value}}",
        "metric": "retrieval",
        "max_gen_toks": 20,
    },
    
    # Code Tasks
    "lcc": {
        "dataset_name": "lcc",
        "prompt": "Complete the following code.\n\n{{context}}\n\n# Complete the code:\n",
        "target": "{{code}}",
        "metric": "code_sim",
        "max_gen_toks": 200,
    },
    "repobench": {
        "dataset_name": "repobench-p",
        "prompt": "Complete the following code based on the repository context.\n\n{{context}}\n\nComplete this function:\n{{function_signature}}\n",
        "target": "{{function_body}}",
        "metric": "code_sim",
        "max_gen_toks": 150,
    },
    "code_debug": {
        "dataset_name": "code_debug",
        "prompt": "Find and fix the bug in the following code.\n\nCode:\n{{context}}\n\nBug description: {{bug_description}}\n\nFixed code:",
        "target": "{{fixed_code}}",
        "metric": "code_sim",
        "max_gen_toks": 200,
    },
    
    # Extended Context Tasks
    "book_qa_eng": {
        "dataset_name": "book_qa_eng",
        "prompt": "Read the book excerpt and answer the question.\n\nBook:\n{{context}}\n\nQuestion: {{question}}\nAnswer:",
        "target": "{{answer}}",
        "metric": "qa_f1",
        "max_gen_toks": 100,
    },
    "paper_assistant": {
        "dataset_name": "paper_assistant",
        "prompt": "You are an academic paper assistant. Based on the paper, answer the question.\n\nPaper:\n{{context}}\n\nQuestion: {{question}}\nAnswer:",
        "target": "{{answer}}",
        "metric": "qa_f1",
        "max_gen_toks": 150,
    },
}

def generate_yaml_config(task_name, config):
    """Generate YAML configuration for a task."""
    yaml_content = {
        "tag": ["longbench_v2"],
        "task": f"longbench_v2_{task_name}",
        "dataset_path": "THUDM/LongBench-v2",
        "dataset_name": config["dataset_name"],
        "output_type": "generate_until",
        "test_split": "test",
        "doc_to_text": config["prompt"],
        "doc_to_target": config["target"],
        "generation_kwargs": {
            "max_gen_toks": config["max_gen_toks"],
            "temperature": 0.1,
            "do_sample": False,
        },
        "process_results": "!function utils.process_results_gen",
        "metric_list": [
            {
                "metric": "score",
                "aggregation": "mean",
                "higher_is_better": True,
            }
        ],
        "metadata": {
            "version": "1.0",
        }
    }
    
    return yaml_content

def main():
    """Generate all configuration files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for task_name, config in TASKS.items():
        yaml_content = generate_yaml_config(task_name, config)
        
        # Write YAML file
        output_path = os.path.join(base_dir, f"{task_name}.yaml")
        with open(output_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        print(f"Generated: {output_path}")
    
    print(f"\nGenerated {len(TASKS)} configuration files for LongBench v2")

if __name__ == "__main__":
    main()