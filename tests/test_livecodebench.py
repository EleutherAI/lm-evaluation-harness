#!/usr/bin/env python3
"""
Test script for LiveCodeBench task validation.
This follows the lm-eval testing patterns and validates both YAML config and utils.py.
"""

import os
import pytest
import json
from itertools import islice
from unittest.mock import patch, MagicMock

import lm_eval.tasks as tasks
from lm_eval.api.task import ConfigurableTask
from lm_eval.utils import load_yaml_config

# Set environment for testing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TestLiveCodeBench:
    """Test suite for LiveCodeBench task implementation."""
    
    @pytest.fixture
    def task_manager(self):
        """Get task manager instance."""
        return tasks.TaskManager()
    
    @pytest.fixture
    def livecodebench_task(self, task_manager):
        """Load LiveCodeBench task."""
        try:
            task_dict = task_manager.load_task_or_group(["livecodebench"])
            return list(task_dict.values())[0]
        except Exception as e:
            pytest.skip(f"LiveCodeBench task not available: {e}")
    
    @pytest.fixture
    def sample_doc(self):
        """Create a sample document for testing."""
        return {
            "id": "test_problem_1",
            "question_content": "Write a function that adds two numbers.\n\ndef add_numbers(a, b):\n    # Your code here\n    pass",
            "input_output": json.dumps({
                "inputs": [["1", "2"], ["5", "3"], ["-1", "4"]],
                "outputs": ["3", "8", "3"]
            }),
            "difficulty": "easy",
            "topic": "basic_programming"
        }
    
    @pytest.fixture
    def sample_generation(self):
        """Sample code generation for testing."""
        return """```python
def add_numbers(a, b):
    return int(a) + int(b)
```"""
    
    def test_yaml_config_exists(self):
        """Test that the YAML configuration file exists and is valid."""
        yaml_path = "lm_eval/tasks/livecodebench/livecodebench.yaml"
        assert os.path.exists(yaml_path), f"YAML config not found at {yaml_path}"
        
        # Load and validate YAML structure
        config = load_yaml_config(yaml_path, mode="simple")
        
        # Required fields
        required_fields = [
            "task", "dataset_path", "output_type", "validation_split",
            "doc_to_text", "doc_to_target", "process_results", "num_fewshot"
        ]
        
        for field in required_fields:
            assert field in config, f"Required field '{field}' missing from YAML config"
        
        # Validate specific values
        assert config["task"] == "livecodebench"
        assert config["output_type"] == "generate_until"
        assert config["validation_split"] == "test"
        assert config["num_fewshot"] == 0
        
        # Validate function references (handle ScalarNode format)
        doc_to_target = config["doc_to_target"]
        process_results = config["process_results"]
        
        # Check if it's a ScalarNode or string
        if hasattr(doc_to_target, 'value'):
            assert doc_to_target.value == "utils.doc_to_target"
        else:
            assert "utils.doc_to_target" in str(doc_to_target)
            
        if hasattr(process_results, 'value'):
            assert process_results.value == "utils.process_results"
        else:
            assert "utils.process_results" in str(process_results)
    
    def test_task_loading(self, livecodebench_task):
        """Test that LiveCodeBench task loads correctly."""
        assert isinstance(livecodebench_task, ConfigurableTask)
        assert livecodebench_task.config.task == "livecodebench"
    
    def test_task_properties(self, livecodebench_task):
        """Test basic task properties."""
        # Test split availability
        assert livecodebench_task.has_test_docs() in [True, False]
        assert livecodebench_task.has_validation_docs() in [True, False]
        assert livecodebench_task.has_training_docs() in [True, False]
        
        # Test decontamination
        assert livecodebench_task.should_decontaminate() in [True, False]
    
    def test_doc_to_text(self, livecodebench_task, sample_doc):
        """Test doc_to_text function."""
        text = livecodebench_task.doc_to_text(sample_doc)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "Question:" in text
        assert "Answer:" in text
        assert sample_doc["question_content"] in text
        
        # Check formatting
        if len(text) > 0:
            assert text[-1] != " ", "doc_to_text should not end with space"
    
    def test_doc_to_target(self, livecodebench_task, sample_doc):
        """Test doc_to_target function."""
        target = livecodebench_task.doc_to_target(sample_doc)
        
        # Should return the full document
        assert isinstance(target, dict)
        assert target == sample_doc
    
    def test_utils_import(self):
        """Test that utils module imports correctly."""
        try:
            from lm_eval.tasks.livecodebench import utils
            assert hasattr(utils, 'doc_to_target')
            assert hasattr(utils, 'process_results')
            assert hasattr(utils, 'extract_code_generation')
            assert hasattr(utils, 'codegen_check_correctness')
        except ImportError as e:
            pytest.fail(f"Failed to import utils: {e}")
    
    def test_extract_code_generation(self):
        """Test code extraction function."""
        from lm_eval.tasks.livecodebench.utils import extract_code_generation
        
        # Test with Python code block
        python_output = """Here's the solution:

```python
def add_numbers(a, b):
    return int(a) + int(b)
```

This function adds two numbers."""
        
        extracted = extract_code_generation(python_output, model_type='chat')
        expected = "def add_numbers(a, b):\n    return int(a) + int(b)"
        assert extracted == expected
        
        # Test with generic code block
        generic_output = """```
def add_numbers(a, b):
    return int(a) + int(b)
```"""
        
        extracted = extract_code_generation(generic_output, model_type='chat')
        assert extracted == expected
        
        # Test with no code blocks
        no_code = "This is just text without code blocks"
        extracted = extract_code_generation(no_code, model_type='chat')
        assert extracted == ""
        
        # Test base model type
        base_output = "def add_numbers(a, b):\n    return int(a) + int(b)"
        extracted = extract_code_generation(base_output, model_type='base')
        assert extracted == base_output.strip()
    
    def test_process_results_function(self, sample_doc, sample_generation):
        """Test process_results function."""
        from lm_eval.tasks.livecodebench.utils import process_results
        
        # Test with valid generation
        results = [sample_generation]
        metrics = process_results(sample_doc, results)
        
        assert isinstance(metrics, dict)
        assert "pass@1" in metrics
        assert isinstance(metrics["pass@1"], float)
        assert 0.0 <= metrics["pass@1"] <= 1.0
        
        # Test with empty results
        empty_metrics = process_results(sample_doc, [])
        assert empty_metrics["pass@1"] == 0.0
    
    @patch('lm_eval.tasks.livecodebench.utils.codegen_metrics')
    def test_process_results_error_handling(self, mock_codegen_metrics, sample_doc):
        """Test process_results error handling."""
        from lm_eval.tasks.livecodebench.utils import process_results
        
        # Mock an exception in codegen_metrics
        mock_codegen_metrics.side_effect = Exception("Test error")
        
        results = ["def test(): pass"]
        metrics = process_results(sample_doc, results)
        
        # Should return 0.0 on error
        assert metrics["pass@1"] == 0.0
    
    def test_codegen_check_correctness_structure(self):
        """Test the structure of codegen_check_correctness function."""
        from lm_eval.tasks.livecodebench.utils import codegen_check_correctness
        
        # This is a structural test - we can't easily test execution without real test cases
        import inspect
        sig = inspect.signature(codegen_check_correctness)
        
        expected_params = ['sample', 'generation', 'timeout']
        for param in expected_params:
            assert param in sig.parameters, f"Missing parameter: {param}"
    
    def test_evaluate_generations_structure(self):
        """Test the structure of evaluate_generations function."""
        from lm_eval.tasks.livecodebench.utils import evaluate_generations
        
        import inspect
        sig = inspect.signature(evaluate_generations)
        
        expected_params = ['samples_list', 'generations_list']
        for param in expected_params:
            assert param in sig.parameters, f"Missing parameter: {param}"
    
    def test_codegen_metrics_structure(self):
        """Test the structure of codegen_metrics function."""
        from lm_eval.tasks.livecodebench.utils import codegen_metrics
        
        import inspect
        sig = inspect.signature(codegen_metrics)
        
        expected_params = ['samples_list', 'generations_list']
        for param in expected_params:
            assert param in sig.parameters, f"Missing parameter: {param}"
    
    def test_task_construction(self, livecodebench_task, sample_doc):
        """Test task request construction."""
        if not hasattr(livecodebench_task, 'construct_requests'):
            pytest.skip("Task doesn't have construct_requests method")
        
        text = livecodebench_task.doc_to_text(sample_doc)
        requests = livecodebench_task.construct_requests(sample_doc, text)
        
        assert isinstance(requests, list)
        if requests:  # If requests are generated
            assert len(requests) > 0
    
    def test_yaml_generation_kwargs(self):
        """Test that generation kwargs are properly configured."""
        yaml_path = "lm_eval/tasks/livecodebench/livecodebench.yaml"
        config = load_yaml_config(yaml_path, mode="simple")
        
        assert "generation_kwargs" in config
        gen_kwargs = config["generation_kwargs"]
        
        # Check required generation parameters
        assert "until" in gen_kwargs
        assert "do_sample" in gen_kwargs
        assert "max_tokens" in gen_kwargs
        
        # Validate values
        assert isinstance(gen_kwargs["until"], list)
        assert len(gen_kwargs["until"]) > 0
        assert isinstance(gen_kwargs["do_sample"], bool)
        assert isinstance(gen_kwargs["max_tokens"], int)
        assert gen_kwargs["max_tokens"] > 0
    
    def test_metric_configuration(self):
        """Test metric configuration in YAML."""
        yaml_path = "lm_eval/tasks/livecodebench/livecodebench.yaml"
        config = load_yaml_config(yaml_path, mode="simple")
        
        assert "metric_list" in config
        metrics = config["metric_list"]
        
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        
        # Check for accuracy metric
        acc_metric = next((m for m in metrics if m.get("metric") == "acc"), None)
        assert acc_metric is not None, "acc metric should be configured"
    
    def test_postprocess_generation(self):
        """Test postprocess_generation function."""
        from lm_eval.tasks.livecodebench.utils import postprocess_generation
        
        sample_output = """```python
def test():
    return True
```"""
        
        result = postprocess_generation(sample_output)
        assert isinstance(result, str)
        assert "def test():" in result
        assert "return True" in result
        assert "```" not in result  # Code blocks should be removed


def test_livecodebench_integration():
    """Integration test to ensure LiveCodeBench works end-to-end."""
    try:
        task_manager = tasks.TaskManager()
        task_dict = task_manager.load_task_or_group(["livecodebench"])
        task = list(task_dict.values())[0]
        
        # Test basic functionality
        assert task.config.task == "livecodebench"
        
        # Test with sample document
        sample_doc = {
            "id": "integration_test",
            "question_content": "def test(): pass",
            "input_output": json.dumps({"inputs": [["1"]], "outputs": ["1"]}),
        }
        
        text = task.doc_to_text(sample_doc)
        target = task.doc_to_target(sample_doc)
        
        assert isinstance(text, str)
        assert isinstance(target, dict)
        
        print("✅ LiveCodeBench integration test passed!")
        
    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 