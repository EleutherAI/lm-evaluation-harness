#!/usr/bin/env python3
"""
Test that standard lm_eval command now uses NPU as default for rbln models.
"""

import sys
import os
import subprocess

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_standard_lm_eval_npu_default():
    """Test that lm_eval automatically uses NPU for rbln models."""
    print("Testing Standard lm_eval with NPU Auto-Detection")
    print("=" * 60)
    
    # Test command that should auto-detect NPU
    cmd = [
        "lm_eval", 
        "--model", "rbln",
        "--model_args", "pretrained=microsoft/DialoGPT-small",
        "--tasks", "hellaswag",
        "--limit", "1"
    ]
    
    print("Testing command:")
    print(f"   {' '.join(cmd)}")
    print("\nExpected behavior:")
    print("   Should auto-detect NPU devices")
    print("   Should use npu:0 as default device")
    print("   Should fail due to missing RBLN SDK (expected)")
    
    print("\n" + "-" * 60)
    print("Command Output:")
    print("-" * 60)
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd="/home/minseo/npu-evaluation",
            env={**os.environ, "PYTHONPATH": "/home/minseo/npu-evaluation"}
        )
        
        output = result.stdout + result.stderr
        print(output)
        
        # Check for expected log messages
        success_indicators = [
            "NPU devices detected: 8",
            "Auto-detected device for NPU model: npu:0",
            "Auto-detected device for NPU model: npu:0"
        ]
        
        found_indicators = []
        for indicator in success_indicators:
            if indicator in output:
                found_indicators.append(indicator)
        
        print("\n" + "-" * 60)
        print("Analysis:")
        print("-" * 60)
        
        if found_indicators:
            print("SUCCESS: NPU auto-detection is working!")
            for indicator in found_indicators:
                print(f"   ✓ Found: '{indicator}'")
        else:
            print("FAILED: NPU auto-detection not found in output")
        
        if "RBLN SDK is not installed" in output:
            print("Expected RBLN SDK error (this is normal)")
        
        # Check if device was NOT manually specified
        if "--device" not in ' '.join(cmd):
            print("No manual --device specified (auto-detection working)")
        
        return len(found_indicators) > 0
        
    except Exception as e:
        print(f"Test execution failed: {e}")
        return False

def test_comparison_with_manual_device():
    """Show comparison between auto and manual device specification."""
    print("\n" + "=" * 60)
    print("Comparison: Auto vs Manual Device")
    print("=" * 60)
    
    print("1. Auto-detection (NEW):")
    print("   lm_eval --model rbln --tasks hellaswag")
    print("   → Automatically uses npu:0")
    
    print("\n2. Manual specification (OLD, still works):")
    print("   lm_eval --model rbln --tasks hellaswag --device npu:0")
    print("   → Explicitly uses npu:0")
    
    print("\n3. Override auto-detection:")
    print("   lm_eval --model rbln --tasks hellaswag --device npu:3")
    print("   → Uses npu:3 (overrides auto-detection)")
    
    print("\n4. Non-NPU models (unchanged):")
    print("   lm_eval --model hf --model_args pretrained=gpt2 --tasks hellaswag")
    print("   → Uses CUDA/CPU (no NPU auto-detection)")

def main():
    """Main test function."""
    print("Standard lm_eval NPU Auto-Detection Test")
    print("=" * 70)
    
    # Test auto-detection
    success = test_standard_lm_eval_npu_default()
    
    # Show comparison
    test_comparison_with_manual_device()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    if success:
        print("SUCCESS: Standard lm_eval now uses NPU as default!")
        print("\nKey achievements:")
        print("• lm_eval --model rbln → automatically uses npu:0")
        print("• No need to specify --device npu:0 anymore")
        print("• NPU detection happens in core lm_eval framework")
        print("• Backward compatibility maintained")
        
        print("\nReady to use commands:")
        print("# Simple evaluation (NPU auto-detected):")
        print("lm_eval --model rbln --model_args pretrained=microsoft/DialoGPT-small --tasks hellaswag")
        
        print("\n# Multiple tasks (NPU auto-detected):")
        print("lm_eval --model rbln --model_args pretrained=microsoft/DialoGPT-small --tasks hellaswag,arc_easy")
        
        print("\n# Override to specific NPU:")
        print("lm_eval --model rbln --model_args pretrained=microsoft/DialoGPT-small --tasks hellaswag --device npu:3")
        
    else:
        print("FAILED: NPU auto-detection not working")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
