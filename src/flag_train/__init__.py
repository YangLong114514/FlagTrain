"""FlagTrain: a compatible reimplementation of DeepSpeed.

The operator layer lives in ``flag_train.deepspeed`` and mirrors the public names
of ``deepspeed.inference.v2.kernels.ragged_ops``, so a caller can swap the import
-- ``from flag_train.deepspeed import blocked_flash`` -- rather than rewrite the
call. The promised names are ``flag_train.deepspeed``'s ``__all__``; the upstream
versions and hardware actually verified are in ``docs/compatibility.md``.
"""

# ``deepspeed`` is imported for the attribute after a bare ``import flag_train``,
# and for its side effect: importing that package is what runs the vendor-operator
# registration that decides which ``blocked_flash`` its names receive. ``testing``
# is a re-export. ``runtime`` is used just below.
from flag_train import deepspeed, runtime, testing  # noqa: F401

device = runtime.device.name
vendor_name = runtime.device.vendor_name


class use_train:
    """Placeholder for FlagTrain's aten-patching context manager.

    The real ``use_train`` patches ``torch.ops.aten`` for the duration of the
    block, which needs an aten-patch registry (``torch.library.Library``) that
    FlagTrain does not carry yet. That is a different thing from the vendor
    operator registry, which already runs on import -- see
    ``flag_train.deepspeed``. Nothing in this repository enters this class -- the
    benchmarks pass an explicit ``gems_op`` -- so it only has to exist and fail
    loudly if a caller assumes the FlagTrain behaviour.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "flag_train has no registered aten operator overrides yet; pass the "
            "operator under test explicitly instead of relying on use_train()."
        )


# The operators are deliberately absent: they are reached through
# ``flag_train.deepspeed``, which is the one place their names are enumerated and
# the place the vendor registration rewrites. This exports only what the top level
# itself defines.
__all__ = [
    "device",
    "use_train",
    "vendor_name",
]
