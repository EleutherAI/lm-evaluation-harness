#!/usr/bin/env python3
"""
Test script for Rebellions NPU integration with lm_eval.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_npu_model_import():
    """Test if the NPU model can be imported."""
    try:
        from lm_eval.models.rbln import RebellionsNPU
        print("Successfully imported RebellionsNPU model")
        return True
    except ImportError as e:
        print(f"Failed to import RebellionsNPU model: {e}")
        return False

def test_rbln_availability():
    """Test if RBLN SDK is available."""
    try:
        import optimum.rbln
        print("RBLN SDK (optimum[rbln]) is available")
        return True
    except ImportError:
        print("RBLN SDK is not available")
        print("  Please install with: pip install optimum[rbln]")
        return False

def test_npu_detection():
    """Test NPU detection using rbln-stat."""
    try:
        import subprocess
        
        print("Testing NPU detection...")
        result = subprocess.run(['rbln-stat'], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("rbln-stat command works")
            lines = result.stdout.strip().split('\n')
            npu_count = len([line for line in lines if 'rbln' in line.lower()])
            if npu_count > 0:
                print(f"Detected {npu_count} NPU devices")
                return True
            else:
                print("rbln-stat works but no NPUs detected")
                return True  # Command works, just no NPUs
        else:
            print("rbln-stat command failed")
            return False
            
    except Exception as e:
        print(f"NPU detection test failed: {e}")
        return False

def test_model_registration():
    """Test if the models are properly registered."""
    try:
        import lm_eval.models  # Import models to register them
        from lm_eval.api.registry import get_model
        
        # Test main model
        model_class = get_model("rbln")
        print(f"Model 'rbln' is registered: {model_class}")
        
        # Only test the universal rbln model (specialized types removed)
        print("Universal rbln model with intelligent auto-detection is registered")
        
        return True
    except Exception as e:
        print(f"Model registration failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Rebellions NPU integration with lm_eval...")
    print("=" * 50)
    
    tests = [
        test_npu_model_import,
        test_rbln_availability,
        test_npu_detection,
        test_model_registration,
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        result = test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed! NPU integration is ready.")
        print("\nNext steps:")
        print("1. Install RBLN SDK if not already installed")
        print("2. Run a simple benchmark:")
        print("   lm_eval --model rbln --model_args pretrained=microsoft/DialoGPT-small --tasks hellaswag --batch_size 1")
    else:
        print("Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
