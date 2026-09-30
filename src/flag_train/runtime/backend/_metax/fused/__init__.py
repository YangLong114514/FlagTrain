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
"""Backend-specialised fused operators.

Empty on purpose, for the reasons in ``ops/__init__.py``: ``fused`` is the second
module ``import_vendor_extra_lib`` looks for, and without it every import reports
that no specialised fused operators were found.

Functions bound here are registered after the ones in ``ops/``, so a name bound in
both is taken from here.
"""

__all__ = []
