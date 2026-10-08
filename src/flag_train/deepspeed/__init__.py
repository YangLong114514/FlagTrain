"""DeepSpeed-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

from .blocked_flash import blocked_flash
from .lamb import lamb
from .wf6af16_linear import pack_weights_fp6, quantize_weights_fp6, wf6af16_linear

__all__ = [
    "blocked_flash",
    "lamb",
    "pack_weights_fp6",
    "quantize_weights_fp6",
    "wf6af16_linear",
]


SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
