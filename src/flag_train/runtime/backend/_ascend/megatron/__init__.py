# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Backend ``megatron`` operators.

Mirrors ``flag_train.megatron``: an override lives in the same-named module as
the generic one it replaces. Define it here, re-export it, and list it in
``__all__`` -- ``SpecOpRegistrar`` registers exactly ``__all__``, overriding the
same-named generic implementation.
"""

from .fast_gelu import fast_gelu

# Bind h_post before fused_gated_norm triggers generic Megatron registration.
from .fused_h_post_bda import fused_h_post_bda_backward, fused_h_post_bda_forward  # isort: skip
from .fused_gated_norm import fused_gated_norm
from .fused_indices_to_multihot import fused_indices_to_multihot
from .fused_pad_routing_map import fused_pad_routing_map
from .fused_proj_rms_compute_h_backward import fused_proj_rms_compute_h_backward
from .fused_proj_rms_compute_h_forward import fused_proj_rms_compute_h_forward
from .scaled_upper_triang_masked_softmax_forward import (
    scaled_upper_triang_masked_softmax_forward,
)
from .weighted_squared_relu_impl import weighted_squared_relu_impl

__all__ = [
    "fast_gelu",
    "fused_gated_norm",
    "fused_h_post_bda_backward",
    "fused_h_post_bda_forward",
    "fused_indices_to_multihot",
    "fused_pad_routing_map",
    "fused_proj_rms_compute_h_backward",
    "fused_proj_rms_compute_h_forward",
    "scaled_upper_triang_masked_softmax_forward",
    "weighted_squared_relu_impl",
]
