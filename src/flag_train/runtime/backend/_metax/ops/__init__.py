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
"""MetaX-specialised operators.

The subpackages here mirror ``flag_train``'s, so which generic module an override
belongs to is readable from the path: an override for
``flag_train.deepspeed.<name>`` lives in ``deepspeed/<name>.py`` and is re-exported
by ``deepspeed/__init__.py`` -- so a new operator is one module beside it plus one
line in its ``__all__``, and this file stays untouched.

``SpecOpRegistrar`` reads this module and every subpackage below it, taking each
package's operators from its ``__all__`` and writing each one over the same-named
generic implementation -- so an operator is registered by listing it in the
``__all__`` of the package that re-exports it.
"""
