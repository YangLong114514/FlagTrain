"""DeepSpeed-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

from .blocked_flash import blocked_flash
from .evoformer_attn import evoformer_attn
from .lamb import lamb
from .mixed_moe_gemm import mixed_moe_gemm
from .packbits import packbits
from .unpackbits import unpackbits
from .wf6af16_linear import wf6af16_linear

__all__ = [
    "blocked_flash",
    "evoformer_attn",
    "lamb",
    "mixed_moe_gemm",
    "packbits",
    "unpackbits",
    "wf6af16_linear",
]


SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
