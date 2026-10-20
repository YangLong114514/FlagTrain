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
"""Iluvatar wf6af16_linear: the generic kernel minus the word-wide loads.

The generic implementation is ``flag_train.deepspeed.wf6af16_linear``. On
Iluvatar's CoreX stack its byte-load path is correct, but the WIDE_LOAD path
(uint32 pointer casts indexed by the K-loop counter) is miscompiled whenever
the K loop iterates more than once: single-iteration shapes pass, while
multi-iteration ones come back with ~37% of the output elements wrong (the
same wrong values regardless of split-K, i.e. corrupted unpacked weights).
That pattern passes on NVIDIA and never runs on MetaX (which has its own
kernel), so the wide path stays enabled in the generic implementation and is
only forced off here.

Everything else -- tile space, split-K choice, reduction kernel -- is the
generic code's; this module only pins the load path.

Registration is by name. This module's ``__init__`` re-exports
``wf6af16_linear``, and ``SpecOpRegistrar`` writes it over the same-named
generic function in ``flag_train.deepspeed`` when that package is imported --
the namespace the tests and the benchmark read the operator from.
"""

import logging

from flag_train.deepspeed.wf6af16_linear import (
    wf6af16_linear as _generic_wf6af16_linear,
)

logger = logging.getLogger(__name__)


def wf6af16_linear(*args, **kwargs):
    """Generic wf6af16_linear with the word-wide weight loads forced off.

    Same contract as ``flag_train.deepspeed.wf6af16_linear``; see the module
    docstring for why ``wide_load`` is pinned to ``False`` on this backend.
    """
    logger.debug("TRAIN WF6AF16_LINEAR (Iluvatar)")
    kwargs["wide_load"] = False
    return _generic_wf6af16_linear(*args, **kwargs)
