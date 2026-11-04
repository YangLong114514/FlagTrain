"""Megatron-related implementations."""

from .bias_swiglu_impl import bias_swiglu_impl
from .fast_gelu import fast_gelu
from .fused_gated_norm import fused_gated_norm
from .fused_h_aggregate_backward import fused_h_aggregate_backward
from .fused_h_aggregate_forward import fused_h_aggregate_forward
from .fused_h_post_bda_backward import fused_h_post_bda_backward
from .fused_h_post_bda_forward import fused_h_post_bda_forward
from .fused_indices_to_multihot import fused_indices_to_multihot
from .fused_pad_routing_map import fused_pad_routing_map
from .fused_proj_rms_backward import fused_proj_rms_backward
from .fused_proj_rms_compute_h_backward import fused_proj_rms_compute_h_backward
from .fused_proj_rms_compute_h_forward import fused_proj_rms_compute_h_forward
from .fused_proj_rms_forward import fused_proj_rms_forward
from .quick_gelu import quick_gelu
from .scaled_masked_softmax_cuda_backward import scaled_masked_softmax_cuda_backward
from .scaled_masked_softmax_cuda_forward import scaled_masked_softmax_cuda_forward
from .scaled_upper_triang_masked_softmax_forward import (
    scaled_upper_triang_masked_softmax_forward,
)
from .weighted_squared_relu_impl import weighted_squared_relu_impl

__all__ = [
    "bias_swiglu_impl",
    "fast_gelu",
    "fused_gated_norm",
    "fused_h_aggregate_backward",
    "fused_h_aggregate_forward",
    "fused_h_post_bda_backward",
    "fused_h_post_bda_forward",
    "fused_indices_to_multihot",
    "fused_pad_routing_map",
    "fused_proj_rms_backward",
    "fused_proj_rms_compute_h_backward",
    "fused_proj_rms_compute_h_forward",
    "fused_proj_rms_forward",
    "quick_gelu",
    "scaled_masked_softmax_cuda_backward",
    "scaled_masked_softmax_cuda_forward",
    "scaled_upper_triang_masked_softmax_forward",
    "weighted_squared_relu_impl",
]

# The vendor-operator registration for these names runs in flag_train/__init__.py
# once this package and its siblings are fully imported -- see the comment there.
