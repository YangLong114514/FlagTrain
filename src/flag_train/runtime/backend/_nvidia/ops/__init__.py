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
``flag_train.deepspeed.<name>`` is defined in ``deepspeed/<name>.py`` and
re-exported by ``deepspeed/__init__.py``.

Empty on purpose: the generic implementations are the NVIDIA ones, so nothing here
needs a different implementation yet. It exists so ``import_vendor_extra_lib``
finds an ``ops`` package rather than printing "no specialized common operators were
found" on every import, and so this backend's layout matches the others'.

This is the vendor-wide layer. An operator that only some NVIDIA generations need
belongs one level down, in ``<arch>/ops/`` beside this file, which
``BackendArchEvent`` loads for the detected architecture.

``SpecOpRegistrar`` reads this module and every subpackage below it, taking each
package's operators from its ``__all__`` and writing each one over the same-named
generic implementation -- so an operator is registered by listing it in the
``__all__`` of the package that re-exports it, and ``__all__`` here is empty
because nothing at this level has a specialised implementation.
"""

__all__ = []
