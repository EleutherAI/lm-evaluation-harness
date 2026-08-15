#!/usr/bin/env python3
"""
Standalone NPU detection script for Rebellions NPU.

This script checks if NPU devices are available and provides detailed information.
"""

import sys
import os

# Add the parent directory to Python path to find lm_eval
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main function."""
    print("Rebellions NPU Detection")
    print("=" * 30)
    
    try:
        from lm_eval.npu_utils.npu_detection import get_npu_info, format_npu_info, is_npu_available, get_npu_count
        
        # Get comprehensive NPU information
        npu_info = get_npu_info()
        
        # Display formatted information
        print(format_npu_info(npu_info))
        
        # Additional details
        print(f"\nDetailed Information:")
        print(f"  NPU Available: {is_npu_available()}")
        print(f"  NPU Count: {get_npu_count()}")
        print(f"  rbln-stat Available: {npu_info['rbln_stat_available']}")
        
        # Show raw info if available
        if npu_info["raw_info"]:
            print(f"\nRaw rbln-stat Output:")
            import json
            print(json.dumps(npu_info["raw_info"], indent=2))
        
        # Provide next steps
        print(f"\nNext Steps:")
        if npu_info["available"]:
            print("NPU devices are available!")
            print("   You can now run lm_eval with NPU models:")
            print("   lm_eval --model rbln --model_args pretrained=microsoft/DialoGPT-small --tasks hellaswag --device npu:0")
        else:
            print("No NPU devices detected.")
            if not npu_info["rbln_stat_available"]:
                print("   - Install RBLN SDK: https://docs.rbln.ai/")
                print("   - Ensure rbln-stat command is available")
            else:
                print("   - Check NPU hardware connection")
                print("   - Verify NPU drivers are installed")
                print("   - Try running 'rbln-stat' manually")
        
        return 0 if npu_info["available"] else 1
        
    except ImportError as e:
        print(f"Error importing NPU detection module: {e}")
        print("   Make sure lm_eval is properly installed")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
