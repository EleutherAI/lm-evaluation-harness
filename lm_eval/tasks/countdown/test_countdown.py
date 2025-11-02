#!/usr/bin/env python3
"""
Test script for the Countdown task implementation.
This tests the task configuration and utilities without requiring full harness installation.
"""

import yaml
import sys
from pathlib import Path

# Import utils from the same directory
import utils


def test_yaml_config():
    """Test that the YAML config is valid."""
    print("=" * 60)
    print("Testing YAML Configuration")
    print("=" * 60)
    
    yaml_path = Path(__file__).parent / "countdown.yaml"
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ YAML loaded successfully")
    print(f"  Task name: {config.get('task')}")
    print(f"  Dataset: {config.get('dataset_path')}")
    print(f"  Output type: {config.get('output_type')}")
    print(f"  Test split: {config.get('test_split')}")
    print(f"  Metrics: {config.get('metric_list')}")
    print()
    
    # Validate required fields
    required = ['task', 'dataset_path', 'output_type', 'metric_list']
    for field in required:
        if field not in config:
            print(f"✗ Missing required field: {field}")
            return False
    
    print(f"✓ All required fields present")
    return True


def test_doc_to_text():
    """Test the doc_to_text function."""
    print("=" * 60)
    print("Testing doc_to_text")
    print("=" * 60)
    
    test_cases = [
        {'nums': [41, 70, 18, 35], 'target': 57},
        {'nums': [7, 24, 4], 'target': 35},
        {'nums': [10, 5, 2], 'target': 17},
    ]
    
    for doc in test_cases:
        text = utils.doc_to_text(doc)
        print(f"Input: nums={doc['nums']}, target={doc['target']}")
        print(f"Output: {text}")
        print()
        
        # Validate output contains the numbers and target
        for num in doc['nums']:
            if str(num) not in text:
                print(f"✗ Number {num} not found in output")
                return False
        if str(doc['target']) not in text:
            print(f"✗ Target {doc['target']} not found in output")
            return False
    
    print(f"✓ doc_to_text working correctly")
    return True


def test_extract_solution():
    """Test the extract_solution function."""
    print("=" * 60)
    print("Testing extract_solution")
    print("=" * 60)
    
    test_cases = [
        ('(70 - 41) + (35 - 18)', '(70 - 41) + (35 - 18)'),
        ('<answer>(70 - 41) + (35 - 18)</answer>', '(70 - 41) + (35 - 18)'),
        ('The answer is: (70 - 41) + (35 - 18) = 57', '(70 - 41) + (35 - 18)'),
        ('70 - 41 + 35 - 18', '70 - 41 + 35 - 18'),
        ('I can solve this: 70 - 41 + 35 - 18', '70 - 41 + 35 - 18'),
        ('', None),
        ('No equation here', None),
    ]
    
    all_passed = True
    for input_str, expected in test_cases:
        extracted = utils.extract_solution(input_str)
        status = "✓" if extracted == expected else "✗"
        print(f"{status} Input: {input_str[:50]}...")
        print(f"  Expected: {expected}")
        print(f"  Got: {extracted}")
        print()
        if extracted != expected:
            all_passed = False
    
    if all_passed:
        print(f"✓ extract_solution working correctly")
    else:
        print(f"✗ Some extract_solution tests failed")
    return all_passed


def test_validate_equation():
    """Test the validate_equation function."""
    print("=" * 60)
    print("Testing validate_equation")
    print("=" * 60)
    
    test_cases = [
        ('(70 - 41) + (35 - 18)', [41, 70, 18, 35], True),
        ('70 - 41 + 35 - 18', [41, 70, 18, 35], True),
        ('70 + 41', [41, 70, 18, 35], False),  # Not all numbers used
        ('70 + 41 + 100', [41, 70, 18, 35], False),  # Wrong number used
        ('70 + 70 + 41', [41, 70, 18, 35], False),  # Duplicate usage
    ]
    
    all_passed = True
    for eq, nums, expected in test_cases:
        result = utils.validate_equation(eq, nums)
        status = "✓" if result == expected else "✗"
        print(f"{status} Equation: {eq}")
        print(f"  Numbers: {nums}")
        print(f"  Expected: {expected}, Got: {result}")
        print()
        if result != expected:
            all_passed = False
    
    if all_passed:
        print(f"✓ validate_equation working correctly")
    else:
        print(f"✗ Some validate_equation tests failed")
    return all_passed


def test_evaluate_equation():
    """Test the evaluate_equation function."""
    print("=" * 60)
    print("Testing evaluate_equation")
    print("=" * 60)
    
    test_cases = [
        ('(70 - 41) + (35 - 18)', 46),
        ('70 - 41 + 35 - 18', 46),
        ('2 + 2', 4),
        ('10 * 5 + 2', 52),
        ('(10 + 5) * 2', 30),
        ('invalid', None),
        ('10 / 0', None),  # Division by zero
    ]
    
    all_passed = True
    for eq, expected in test_cases:
        result = utils.evaluate_equation(eq)
        status = "✓" if result == expected else "✗"
        print(f"{status} Equation: {eq}")
        print(f"  Expected: {expected}, Got: {result}")
        print()
        if result != expected:
            all_passed = False
    
    if all_passed:
        print(f"✓ evaluate_equation working correctly")
    else:
        print(f"✗ Some evaluate_equation tests failed")
    return all_passed


def test_compute_score():
    """Test the compute_score function."""
    print("=" * 60)
    print("Testing compute_score")
    print("=" * 60)
    
    test_cases = [
        # (solution_str, ground_truth, expected_score)
        ('<answer>(70 - 41) + (35 - 18)</answer>', 
         {'target': 46, 'numbers': [41, 70, 18, 35]}, 1.0),
        ('70 - 41 + 35 - 18', 
         {'target': 46, 'numbers': [41, 70, 18, 35]}, 1.0),
        ('<answer>(70 - 41) + (35 - 18)</answer>', 
         {'target': 50, 'numbers': [41, 70, 18, 35]}, 0.1),  # Wrong target
        ('70 + 41', 
         {'target': 111, 'numbers': [41, 70, 18, 35]}, 0.1),  # Not all numbers used
        ('no equation here', 
         {'target': 46, 'numbers': [41, 70, 18, 35]}, 0.0),  # No valid equation
    ]
    
    all_passed = True
    for sol, gt, expected in test_cases:
        result = utils.compute_score(sol, gt)
        status = "✓" if abs(result - expected) < 0.01 else "✗"
        print(f"{status} Solution: {sol[:50]}...")
        print(f"  Ground truth: {gt}")
        print(f"  Expected score: {expected}, Got: {result}")
        print()
        if abs(result - expected) >= 0.01:
            all_passed = False
    
    if all_passed:
        print(f"✓ compute_score working correctly")
    else:
        print(f"✗ Some compute_score tests failed")
    return all_passed


def test_process_results():
    """Test the process_results function."""
    print("=" * 60)
    print("Testing process_results")
    print("=" * 60)
    
    doc = {'nums': [41, 70, 18, 35], 'target': 46}
    results = ['<answer>(70 - 41) + (35 - 18)</answer>']
    
    output = utils.process_results(doc, results)
    print(f"Input doc: {doc}")
    print(f"Results: {results}")
    print(f"Output: {output}")
    print()
    
    if 'countdown_score' in output:
        print(f"✓ process_results working correctly")
        print(f"  Score: {output['countdown_score']}")
        return True
    else:
        print(f"✗ process_results missing 'countdown_score' key")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("COUNTDOWN TASK IMPLEMENTATION TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        ("YAML Configuration", test_yaml_config),
        ("doc_to_text", test_doc_to_text),
        ("extract_solution", test_extract_solution),
        ("validate_equation", test_validate_equation),
        ("evaluate_equation", test_evaluate_equation),
        ("compute_score", test_compute_score),
        ("process_results", test_process_results),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ {name} raised an exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print()
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

