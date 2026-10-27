"""Megatron-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

from .fused_h_post_bda import fused_h_post_bda_backward, fused_h_post_bda_forward
from .quick_gelu import quick_gelu

__all__ = [
    "fused_h_post_bda_backward",
    "fused_h_post_bda_forward",
    "quick_gelu",
]


SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
