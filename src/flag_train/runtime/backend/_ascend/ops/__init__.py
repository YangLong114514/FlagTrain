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
"""Backend-specialised operators.

The subpackages here mirror ``flag_train``'s, so which generic module an override
belongs to is readable from the path: an override for
``flag_train.deepspeed.<name>`` is defined in ``deepspeed/<name>.py``, re-exported
by ``deepspeed/__init__.py`` and then by this file -- the module
``SpecOpRegistrar`` reads.

Empty on purpose: no operator in this backend needs an implementation of its own
yet. It exists so ``import_vendor_extra_lib`` finds an ``ops`` package rather than
printing "no specialized common operators were found" on every import, and so this
backend's layout matches the others'.

``SpecOpRegistrar`` collects this module's functions by name --
``inspect.getmembers(..., inspect.isfunction)``, with no filtering -- and writes
each one over the same-named generic implementation, so re-export the operators
and nothing else: any other function bound here would be registered as an operator
too.
"""

__all__ = []
