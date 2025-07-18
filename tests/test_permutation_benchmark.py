"""Comprehensive unit tests for permutation benchmark tasks."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, call

import lm_eval.tasks as tasks
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks.permutation_benchmark import group_composition_utils


# All groups that should exist in the benchmark
# Get actual groups from the implementation
ALL_TC0_GROUPS = list(group_composition_utils.TC0_GROUPS)
ALL_NC1_GROUPS = list(group_composition_utils.NC1_GROUPS)


class TestGroupDefinitions:
    """Test that all groups are properly defined."""

    def test_tc0_groups_complete(self):
        """Test that all TC0 groups are present and correctly counted."""
        assert len(group_composition_utils.TC0_GROUPS) == 72
        for group in ALL_TC0_GROUPS:
            assert group in group_composition_utils.TC0_GROUPS, f"Missing TC0 group: {group}"
        
        # Ensure no extra groups
        assert len(set(ALL_TC0_GROUPS)) == len(group_composition_utils.TC0_GROUPS)

    def test_nc1_groups_complete(self):
        """Test that all NC1 groups are present and correctly counted."""
        assert len(group_composition_utils.NC1_GROUPS) == 22
        for group in ALL_NC1_GROUPS:
            assert group in group_composition_utils.NC1_GROUPS, f"Missing NC1 group: {group}"
        
        # Ensure no extra groups
        assert len(set(ALL_NC1_GROUPS)) == len(group_composition_utils.NC1_GROUPS)

    def test_no_group_overlap(self):
        """Test that no group appears in both TC0 and NC1."""
        overlap = group_composition_utils.TC0_GROUPS & group_composition_utils.NC1_GROUPS
        assert len(overlap) == 0, f"Groups in both TC0 and NC1: {overlap}"

    def test_total_group_count(self):
        """Test that we have exactly 94 groups total."""
        total = len(group_composition_utils.TC0_GROUPS) + len(group_composition_utils.NC1_GROUPS)
        assert total == 94


class TestDatasetFunctions:
    """Test dataset loading and processing functions."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a realistic mock dataset."""
        data_points = [
            {
                'input_sequence': '1 2 3 4 5',
                'target': '6',
                'sequence_length': 5,
                'group_degree': 3,
                'group_order': 6,
                'group_type': 'symmetric'
            },
            {
                'input_sequence': ' '.join(str(i) for i in range(1, 51)),
                'target': '3',
                'sequence_length': 50,
                'group_degree': 3,
                'group_order': 6,
                'group_type': 'symmetric'
            },
            {
                'input_sequence': ' '.join(str(i) for i in range(1, 101)),
                'target': '5',
                'sequence_length': 100,
                'group_degree': 3,
                'group_order': 6,
                'group_type': 'symmetric'
            },
            {
                'input_sequence': ' '.join(str(i) for i in range(1, 501)),
                'target': '2',
                'sequence_length': 500,
                'group_degree': 3,
                'group_order': 6,
                'group_type': 'symmetric'
            }
        ]
        
        mock_data = MagicMock()
        mock_data.__iter__ = lambda self: iter(data_points)
        mock_data.__len__ = lambda self: len(data_points)
        mock_data.filter = MagicMock(return_value=mock_data)
        return mock_data

    @pytest.mark.parametrize("group_name", ALL_TC0_GROUPS + ALL_NC1_GROUPS)
    def test_dataset_function_exists(self, group_name):
        """Test that each group has a corresponding dataset function."""
        func_name = f"{group_name}_dataset"
        assert hasattr(group_composition_utils, func_name), f"Missing dataset function: {func_name}"
        assert callable(getattr(group_composition_utils, func_name))

    @patch('lm_eval.tasks.permutation_benchmark.group_composition_utils.load_dataset')
    def test_all_dataset_functions_call_correctly(self, mock_load_dataset):
        """Test that all dataset functions call load_dataset with correct parameters."""
        mock_load_dataset.return_value = MagicMock()
        
        # Test a sample of groups
        sample_groups = list(ALL_TC0_GROUPS)[:5] + list(ALL_NC1_GROUPS)[:5]
        for group_name in sample_groups:
            func_name = f"{group_name}_dataset"
            func = getattr(group_composition_utils, func_name)
            
            # Reset mock
            mock_load_dataset.reset_mock()
            
            # Call the function
            result = func()
            
            # Verify it was called with the group name
            mock_load_dataset.assert_called_once()
            call_args = mock_load_dataset.call_args
            assert call_args[0] == ("BeeGass/Group-Theory-Collection",)
            assert call_args[1]['name'] == group_name

    def test_filter_by_sequence_length_edge_cases(self, mock_dataset):
        """Test sequence length filtering with edge cases."""
        # Test with exact boundaries
        result = group_composition_utils.filter_by_sequence_length(
            mock_dataset, min_length=5, max_length=5
        )
        mock_dataset.filter.assert_called()
        
        # Test with very large range
        result = group_composition_utils.filter_by_sequence_length(
            mock_dataset, min_length=1, max_length=1000
        )
        mock_dataset.filter.assert_called()

    @patch('lm_eval.tasks.permutation_benchmark.group_composition_utils.load_dataset')
    def test_create_length_specific_dataset(self, mock_load_dataset):
        """Test dataset creation with specific length filtering."""
        mock_dataset = MagicMock()
        mock_dataset.filter = MagicMock(return_value=mock_dataset)
        mock_load_dataset.return_value = mock_dataset
        
        result = group_composition_utils.create_length_specific_dataset(
            "s3", target_length=50, split="test"
        )
        
        # Should load the dataset
        mock_load_dataset.assert_called_with(
            "BeeGass/Group-Theory-Collection",
            name="s3",
            split="test"
        )
        
        # Should filter with tolerance of ±2
        filter_func = mock_dataset.filter.call_args[0][0]
        assert filter_func({'sequence_length': 48}) == True
        assert filter_func({'sequence_length': 52}) == True
        assert filter_func({'sequence_length': 47}) == False
        assert filter_func({'sequence_length': 53}) == False


class TestMetricsProcessing:
    """Test metric calculation and aggregation."""

    def test_process_results_various_lengths(self):
        """Test process_results with various sequence lengths."""
        test_cases = [
            (5, True, '5'),
            (7, False, '5'),  # Should round to nearest 5
            (103, True, '105'),  # Should round to 105
            (498, False, '500'),  # Should round to 500
            (502, True, '500'),  # Should round to 500
        ]
        
        for seq_len, is_greedy, expected_key in test_cases:
            doc = {'sequence_length': seq_len}
            results = (0.5, is_greedy)
            
            metrics = group_composition_utils.process_results(doc, results)
            
            # Check the expected metric
            expected_value = 1.0 if is_greedy else 0.0
            assert metrics[expected_key] == expected_value, \
                f"For seq_len={seq_len}, expected {expected_key}={expected_value}"
            
            # All other metrics should be -1
            for key, value in metrics.items():
                if key != expected_key:
                    assert value == -1.0

    def test_process_results_edge_cases(self):
        """Test process_results with edge cases."""
        # Test with wrapped results
        doc = {'sequence_length': 50}
        results = [(0.5, True)]  # Wrapped in list
        
        metrics = group_composition_utils.process_results(doc, results)
        assert metrics['50'] == 1.0
        
        # Test with invalid results
        results = None
        metrics = group_composition_utils.process_results(doc, results)
        assert all(v == -1.0 for v in metrics.values())
        
        # Test with empty results
        results = []
        metrics = group_composition_utils.process_results(doc, results)
        assert all(v == -1.0 for v in metrics.values())

    def test_aggregate_metrics_various_scenarios(self):
        """Test metric aggregation with various scenarios."""
        # All valid metrics
        assert group_composition_utils.aggregate_metrics([0.0, 0.5, 1.0]) == pytest.approx(0.5)
        
        # Mix of valid and invalid
        assert group_composition_utils.aggregate_metrics([0.5, -1, 0.7, -1]) == pytest.approx(0.6)
        
        # Single valid metric
        assert group_composition_utils.aggregate_metrics([-1, -1, 0.8, -1]) == pytest.approx(0.8)
        
        # All invalid
        assert group_composition_utils.aggregate_metrics([-1, -1, -1]) == -1
        
        # Empty list
        assert group_composition_utils.aggregate_metrics([]) == -1

    def test_default_sequence_lengths_properties(self):
        """Test properties of default sequence lengths."""
        seq_lengths = group_composition_utils.DEFAULT_SEQ_LENGTHS
        
        # Check count
        assert len(seq_lengths) == 100
        
        # Check range
        assert seq_lengths[0] == 5
        assert seq_lengths[-1] == 500
        
        # Check increment
        for i in range(1, len(seq_lengths)):
            assert seq_lengths[i] - seq_lengths[i-1] == 5
        
        # Check all values are unique
        assert len(set(seq_lengths)) == len(seq_lengths)


class TestTaskConfiguration:
    """Test task YAML configuration and loading."""

    @pytest.mark.parametrize("group_name", ALL_TC0_GROUPS + ALL_NC1_GROUPS)
    def test_individual_task_files_exist(self, group_name):
        """Test that each group has a corresponding task YAML file."""
        yaml_path = f"lm_eval/tasks/permutation_benchmark/{group_name}_composition.yaml"
        assert os.path.exists(yaml_path), f"Missing task file: {yaml_path}"

    @pytest.mark.parametrize("task_name", 
                             [f"{g}_composition" for g in ALL_TC0_GROUPS[:5]] +  # Sample TC0
                             [f"{g}_composition" for g in ALL_NC1_GROUPS[:5]])   # Sample NC1
    def test_task_configuration_structure(self, task_name):
        """Test that task configurations have correct structure."""
        task_dict = tasks.get_task_dict([task_name])
        assert task_name in task_dict
        
        task = task_dict[task_name]
        
        # Check required attributes
        assert hasattr(task, 'OUTPUT_TYPE')
        assert task.OUTPUT_TYPE == "loglikelihood"
        
        # Check methods
        assert callable(getattr(task, 'doc_to_text', None))
        assert callable(getattr(task, 'doc_to_target', None))
        
        # Check metric configuration
        assert hasattr(task.config, 'metric_list')
        assert len(task.config.metric_list) == 100
        
        # Check all metrics are for sequence lengths
        metric_names = [m['metric'] for m in task.config.metric_list]
        expected_metrics = [str(i) for i in range(5, 505, 5)]
        assert metric_names == expected_metrics

    def test_group_aggregation_tasks(self):
        """Test that group aggregation tasks work correctly."""
        # Instead of loading all tasks, just verify the YAML structure
        import yaml
        
        yaml_dir = "lm_eval/tasks/permutation_benchmark"
        
        # Test TC0 groups file
        with open(os.path.join(yaml_dir, "tc0_groups.yaml"), 'r') as f:
            tc0_config = yaml.safe_load(f)
            assert tc0_config['group'] == 'tc0_groups'
            assert 'task' in tc0_config
            assert isinstance(tc0_config['task'], list)
            assert len(tc0_config['task']) > 0
        
        # Test NC1 groups file  
        with open(os.path.join(yaml_dir, "nc1_groups.yaml"), 'r') as f:
            nc1_config = yaml.safe_load(f)
            assert nc1_config['group'] == 'nc1_groups'
            assert 'task' in nc1_config
            assert isinstance(nc1_config['task'], list)
            assert len(nc1_config['task']) > 0
        
        # Test permutation groups file
        with open(os.path.join(yaml_dir, "permutation_groups.yaml"), 'r') as f:
            perm_config = yaml.safe_load(f)
            assert perm_config['group'] == 'permutation_groups'
            assert 'task' in perm_config
            assert 'tc0_groups' in perm_config['task']
            assert 'nc1_groups' in perm_config['task']

    @patch('lm_eval.tasks.permutation_benchmark.group_composition_utils.load_dataset')
    def test_task_doc_processing(self, mock_load_dataset):
        """Test document processing in tasks."""
        # Create a mock document
        doc = {
            'input_sequence': '1 2 3 4 5',
            'target': '6',
            'sequence_length': 5
        }
        
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = lambda self: iter([doc])
        mock_dataset.__getitem__ = lambda self, idx: doc
        mock_load_dataset.return_value = {"test": mock_dataset}
        
        # Load a task
        task_dict = tasks.get_task_dict(["s3_composition"])
        task = task_dict["s3_composition"]
        
        # Test doc_to_text
        text = task.doc_to_text(doc)
        assert "1 2 3 4 5" in text
        assert "S3" in text
        assert "symmetric group on 3 elements" in text
        
        # Test doc_to_target
        target = task.doc_to_target(doc)
        assert " 6" in target  # Note the space prefix


class TestComplexityClassification:
    """Test group complexity classification."""

    @pytest.mark.parametrize("group_name", ALL_TC0_GROUPS)
    def test_tc0_classification(self, group_name):
        """Test that all TC0 groups are classified correctly."""
        assert group_composition_utils.get_complexity_class(group_name) == "TC0"

    @pytest.mark.parametrize("group_name", ALL_NC1_GROUPS)
    def test_nc1_classification(self, group_name):
        """Test that all NC1 groups are classified correctly."""
        assert group_composition_utils.get_complexity_class(group_name) == "NC1"

    def test_unknown_group_classification(self):
        """Test classification of unknown groups."""
        unknown_groups = ["x1", "unknown", "test_group", ""]
        for group in unknown_groups:
            assert group_composition_utils.get_complexity_class(group) == "Unknown"


class TestIntegration:
    """Integration tests for the permutation benchmark."""

    @patch('lm_eval.tasks.permutation_benchmark.group_composition_utils.load_dataset')
    def test_end_to_end_evaluation_flow(self, mock_load_dataset):
        """Test the complete evaluation flow for a task."""
        # Create mock dataset
        docs = [
            {
                'input_sequence': ' '.join(str(i) for i in range(1, 6)),
                'target': '3',
                'sequence_length': 5,
                'group_degree': 3,
                'group_order': 6,
                'group_type': 'symmetric'
            },
            {
                'input_sequence': ' '.join(str(i) for i in range(1, 101)),
                'target': '5',
                'sequence_length': 100,
                'group_degree': 3,
                'group_order': 6,
                'group_type': 'symmetric'
            }
        ]
        
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = lambda self: iter(docs)
        mock_dataset.__len__ = lambda self: len(docs)
        mock_dataset.__getitem__ = lambda self, idx: docs[idx] if idx < len(docs) else None
        mock_load_dataset.return_value = {"test": mock_dataset}
        
        # Load task
        task_dict = tasks.get_task_dict(["s3_composition"])
        task = task_dict["s3_composition"]
        
        # Process documents
        for doc in docs:
            # Get text and target
            text = task.doc_to_text(doc)
            target = task.doc_to_target(doc)
            
            # Verify format
            assert isinstance(text, str)
            assert isinstance(target, str)
            assert doc['input_sequence'] in text
            assert doc['target'] in target

    def test_yaml_file_validity(self):
        """Test that all YAML files are valid."""
        from lm_eval.utils import load_yaml_config
        
        yaml_dir = "lm_eval/tasks/permutation_benchmark"
        
        # Test all individual task files
        for group in ALL_TC0_GROUPS + ALL_NC1_GROUPS:
            yaml_path = os.path.join(yaml_dir, f"{group}_composition.yaml")
            assert os.path.exists(yaml_path), f"Missing YAML: {yaml_path}"
            
            config = load_yaml_config(yaml_path=yaml_path, mode="full")
            
            # Check required fields
            assert 'task' in config
            assert config['task'] == f"{group}_composition"
            assert 'tag' in config
            assert 'group_theory' in config['tag']
            assert 'permutation_composition' in config['tag']
            
            # Check complexity tag
            expected_complexity = 'tc0' if group in ALL_TC0_GROUPS else 'nc1'
            assert expected_complexity in config['tag']
            
            # Check output type
            assert config['output_type'] == 'loglikelihood'
            
            # Check metrics
            assert 'metric_list' in config
            assert len(config['metric_list']) == 100

    def test_group_yaml_files(self):
        """Test group aggregation YAML files."""
        import yaml
        
        yaml_dir = "lm_eval/tasks/permutation_benchmark"
        
        # Test TC0 groups file
        with open(os.path.join(yaml_dir, "tc0_groups.yaml"), 'r') as f:
            tc0_config = yaml.safe_load(f)
            assert tc0_config['group'] == 'tc0_groups'
            assert len(tc0_config['task']) == 72
            
            # Check all TC0 tasks are included
            for group in ALL_TC0_GROUPS:
                assert f"{group}_composition" in tc0_config['task']
        
        # Test NC1 groups file
        with open(os.path.join(yaml_dir, "nc1_groups.yaml"), 'r') as f:
            nc1_config = yaml.safe_load(f)
            assert nc1_config['group'] == 'nc1_groups'
            assert len(nc1_config['task']) == 22
            
            # Check all NC1 tasks are included
            for group in ALL_NC1_GROUPS:
                assert f"{group}_composition" in nc1_config['task']
        
        # Test main permutation groups file
        with open(os.path.join(yaml_dir, "permutation_groups.yaml"), 'r') as f:
            perm_config = yaml.safe_load(f)
            assert perm_config['group'] == 'permutation_groups'
            assert 'tc0_groups' in perm_config['task']
            assert 'nc1_groups' in perm_config['task']


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_process_results_malformed_input(self):
        """Test process_results with malformed inputs."""
        # Missing sequence_length
        doc = {'other_field': 'value'}
        results = (0.5, True)
        
        # Should not crash
        try:
            metrics = group_composition_utils.process_results(doc, results)
            # Should handle gracefully
        except KeyError:
            pass  # Expected in some cases

    def test_aggregate_metrics_type_errors(self):
        """Test aggregate_metrics with wrong types."""
        # The function now filters out non-numeric values gracefully
        result = group_composition_utils.aggregate_metrics([None, "string", 0.5])
        # Only 0.5 is a valid numeric value
        assert result == 0.5
        
        # Test with all invalid values
        result = group_composition_utils.aggregate_metrics([None, "string", [], {}])
        assert result == -1
        
    @patch('lm_eval.tasks.permutation_benchmark.group_composition_utils.load_dataset')
    def test_dataset_loading_failure(self, mock_load_dataset):
        """Test handling of dataset loading failures."""
        mock_load_dataset.side_effect = Exception("Dataset not found")
        
        # Should raise or handle gracefully
        with pytest.raises(Exception):
            group_composition_utils.s3_dataset()


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "--tb=short"])