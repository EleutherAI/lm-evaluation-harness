"""NPU utilities for Rebellions NPU integration."""

from .npu_detection import (
    is_npu_available,
    get_npu_count,
    detect_npu_devices,
    format_npu_info,
    run_rbln_stat
)

__all__ = [
    'is_npu_available',
    'get_npu_count', 
    'detect_npu_devices',
    'format_npu_info',
    'run_rbln_stat'
]
