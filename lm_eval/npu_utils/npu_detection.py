#!/usr/bin/env python3
"""
NPU Detection utilities for Rebellions NPU.

This module provides functions to detect and get information about
Rebellions NPU hardware using the rbln-stat command.
"""

import subprocess
import json
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def run_rbln_stat() -> Optional[Dict]:
    """
    Run rbln-stat command and return the parsed output.
    
    Returns:
        Dict containing NPU information or None if command fails
    """
    try:
        result = subprocess.run(
            ['rbln-stat'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # Try to parse as JSON first
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                # If not JSON, return raw output
                return {"raw_output": result.stdout.strip()}
        else:
            logger.error(f"rbln-stat failed with return code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("rbln-stat command timed out")
        return None
    except FileNotFoundError:
        logger.error("rbln-stat command not found")
        return None
    except Exception as e:
        logger.error(f"Error running rbln-stat: {e}")
        return None


def detect_npu_devices() -> List[Dict]:
    """
    Detect available NPU devices.
    
    Returns:
        List of dictionaries containing device information
    """
    npu_info = run_rbln_stat()
    if not npu_info:
        return []
    
    devices = []
    
    # Handle different output formats
    if "raw_output" in npu_info:
        # Parse raw text output from rbln-stat
        lines = npu_info["raw_output"].split('\n')
        
        # Look for device table rows (containing | NPU | Name | Device | etc.)
        for line in lines:
            line = line.strip()
            if line.startswith('|') and '| NPU |' not in line and '| N/A |' not in line and '|' in line[1:]:
                # Parse device information
                parts = [part.strip() for part in line.split('|') if part.strip()]
                if len(parts) >= 8:  # Should have NPU, Name, Device, PCI BUS ID, Temp, Power, Perf, Memory, Util
                    try:
                        device_info = {
                            "npu_id": parts[0],
                            "name": parts[1],
                            "device": parts[2],
                            "pci_bus_id": parts[3],
                            "temperature": parts[4],
                            "power": parts[5],
                            "performance": parts[6],
                            "memory": parts[7],
                            "utilization": parts[8] if len(parts) > 8 else "N/A"
                        }
                        devices.append(device_info)
                    except (IndexError, ValueError):
                        # If parsing fails, add as raw info
                        devices.append({"info": line})
    else:
        # Handle JSON output
        if "devices" in npu_info:
            devices = npu_info["devices"]
        elif "npu" in npu_info:
            devices = [npu_info["npu"]]
        else:
            # If we have info but no clear device structure, return the whole thing
            devices = [npu_info]
    
    return devices


def get_npu_count() -> int:
    """
    Get the number of available NPU devices.
    
    Returns:
        Number of NPU devices available
    """
    devices = detect_npu_devices()
    return len(devices)


def is_npu_available() -> bool:
    """
    Check if any NPU devices are available.
    
    Returns:
        True if NPU devices are available, False otherwise
    """
    return get_npu_count() > 0


def get_npu_info() -> Dict:
    """
    Get comprehensive NPU information.
    
    Returns:
        Dictionary containing NPU status and information
    """
    npu_info = run_rbln_stat()
    devices = detect_npu_devices()
    
    return {
        "available": is_npu_available(),
        "count": get_npu_count(),
        "devices": devices,
        "raw_info": npu_info,
        "rbln_stat_available": npu_info is not None
    }


def format_npu_info(info: Dict) -> str:
    """
    Format NPU information for display.
    
    Args:
        info: NPU information dictionary
        
    Returns:
        Formatted string with NPU information
    """
    if not info["rbln_stat_available"]:
        return "rbln-stat command not available or failed"
    
    if not info["available"]:
        return " No NPU devices detected"
    
    output = f"NPU devices detected: {info['count']}\n"
    
    for device in info["devices"]:
        if isinstance(device, dict):
            # Check if it's a parsed device with structured info
            if "npu_id" in device:
                output += f"  NPU {device['npu_id']}:\n"
                output += f"    Name: {device['name']}\n"
                output += f"    Device: {device['device']}\n"
                output += f"    PCI Bus ID: {device['pci_bus_id']}\n"
                output += f"    Temperature: {device['temperature']}\n"
                output += f"    Power: {device['power']}\n"
                output += f"    Performance: {device['performance']}\n"
                output += f"    Memory: {device['memory']}\n"
                output += f"    Utilization: {device['utilization']}\n"
            else:
                # Generic device info
                for key, value in device.items():
                    output += f"    {key}: {value}\n"
        else:
            output += f"    {device}\n"
    
    return output.strip()


def main():
    """Main function for testing NPU detection."""
    print("Rebellions NPU Detection Test")
    print("=" * 40)
    
    # Test NPU detection
    info = get_npu_info()
    print(format_npu_info(info))
    
    # Test individual functions
    print(f"\nNPU Available: {is_npu_available()}")
    print(f"NPU Count: {get_npu_count()}")
    
    # Show raw info if available
    if info["raw_info"]:
        print(f"\nRaw rbln-stat output:")
        print(json.dumps(info["raw_info"], indent=2))


if __name__ == "__main__":
    main()
