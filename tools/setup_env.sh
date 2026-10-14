#!/bin/bash

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

# Prepare one CI job's environment for a backend: apply the backend's
# environment, then install FlagTrain itself and the test dependencies.
#
# Usage: bash tools/setup_env.sh <backend> [--prebuilt|--no-prebuilt]
#
#   --prebuilt     (default) the vendor runtime image already ships torch and
#                  the triton stack, so only FlagTrain and the test deps are
#                  installed. Never install a vendor stack next to the one the
#                  image was built with.
#   --no-prebuilt  also install this backend's `deps:` from backends.yaml,
#                  through the vendor's package index -- for a vendor image
#                  that does not carry the FlagOS stack (thead's own pytorch
#                  image).
#
# Everything comes out of backends.yaml, the same file the nvidia runner image
# documents its pins in:
#   env:         exported through $GITHUB_ENV, so every later step sees it;
#   env_source:  vendor scripts sourced via tools/source_env_bash.sh, with only
#                the variables they changed exported (some vendor scripts
#                self-locate through ${BASH_SOURCE[0]} and misbehave when
#                sourced by a non-bash shell);
#   deps:        the --no-prebuilt install list.
# Outside CI (no $GITHUB_ENV) the same lines are printed as `export K=V`, so
# `eval "$(bash tools/setup_env.sh <backend>)"` works locally.
#
# The trailing import check is deliberate: `import flag_train` needs the
# platform's torch and triton backends, so a broken image or a wrong vendor
# shows up here rather than as a collection error inside pytest.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKENDS_YAML="${REPO_ROOT}/backends.yaml"

usage() {
    echo "usage: setup_env.sh <backend> [--prebuilt|--no-prebuilt]" >&2
    echo "known backends: $(python3 -c "import yaml,sys; print(' '.join(sorted(yaml.safe_load(open(sys.argv[1]))['backends'])))" "${BACKENDS_YAML}" 2>/dev/null || echo '<backends.yaml unreadable>')" >&2
    exit 2
}

BACKEND="${1:-}"
[ -n "${BACKEND}" ] || usage

PREBUILT=1
case "${2:-}" in
    "" | --prebuilt) PREBUILT=1 ;;
    --no-prebuilt) PREBUILT=0 ;;
    *) usage ;;
esac

emit_env() {
    # KEY=VALUE: into $GITHUB_ENV in CI, printed for local `eval`.
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "$1" >>"${GITHUB_ENV}"
    else
        echo "export $1"
    fi
}

FLAGOS_PYPI=""
MIRROR=""
DEPS=""
COMPILER_PKG=""
ENV_SOURCES=""

while IFS=$'\t' read -r kind a b; do
    case "${kind}" in
        PYPI) FLAGOS_PYPI="${a}" ;;
        MIRROR) MIRROR="${a}" ;;
        DEP) DEPS="${DEPS} ${a}" ;;
        COMPILER) COMPILER_PKG="${a}" ;;
        SOURCE) ENV_SOURCES="${ENV_SOURCES} ${a}" ;;
        ENV)
            echo "backend env ${a}=${b}"
            emit_env "${a}=${b}"
            ;;
    esac
done < <(python3 - "${BACKENDS_YAML}" "${BACKEND}" <<'PY'
import os
import re
import sys

import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
backends = cfg["backends"]
key = sys.argv[2]
if key not in backends:
    sys.exit(
        f"tools/setup_env.sh: unknown backend {key!r}; "
        f"known: {', '.join(sorted(backends))}"
    )

backend = backends[key]
# Same rule as FlagGems: the vendor is everything before the last '-'.
vendor = key.rsplit("-", 1)[0] if "-" in key else key

pypi_base = cfg.get("pypi_base", "")
if pypi_base:
    print("PYPI\t" + pypi_base.replace("{vendor}", vendor))
if cfg.get("mirror"):
    print("MIRROR\t" + cfg["mirror"])
for dep in backend.get("deps", []):
    print("DEP\t" + dep)
# FlagTree is the preferred compiler where a backend has one; otherwise the
# backend's Triton package. Installed only with --no-prebuilt, since a
# prebuilt image already carries one of them and installing the other would
# put a second `triton` module next to it.
compiler_pkg = backend.get("flagtree") or backend.get("triton")
if compiler_pkg:
    print("COMPILER\t" + compiler_pkg)
for source in backend.get("env_source", []):
    print("SOURCE\t" + source)
# $VARs are expanded against the running environment plus whatever this same
# entry already defined (iluvatar's PATH refers to its own COREX_ROOT). A value
# that still holds a $VAR after that would be written to $GITHUB_ENV literally
# -- that file does no expansion -- so it is dropped with a warning instead of
# corrupting PATH.
def expand(value, env):
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
    return pattern.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), value)


expanded_env = dict(os.environ)
for name, value in (backend.get("env") or {}).items():
    expanded = expand(value, expanded_env)
    if "$" in expanded:
        print(
            f"tools/setup_env.sh: {name} still has an unexpanded variable "
            f"({value!r}); leaving it to the image",
            file=sys.stderr,
        )
        continue
    expanded_env[name] = expanded
    print(f"ENV\t{name}\t{expanded}")
PY
)

for script in ${ENV_SOURCES}; do
    if [ ! -f "${script}" ]; then
        echo "env_source ${script} is not present in this image; skipping"
        continue
    fi
    echo "sourcing ${script}"
    while IFS= read -r line; do
        case "${line}" in
            "export "*) emit_env "${line#export }" ;;
        esac
    done < <(bash "${REPO_ROOT}/tools/source_env_bash.sh" "${script}")
done

pip_args=(--no-cache-dir)
if [ -n "${FLAGOS_PYPI}" ]; then
    pip_args+=(--index-url "${FLAGOS_PYPI}")
fi
if [ -n "${MIRROR}" ]; then
    pip_args+=(--extra-index-url "${MIRROR}")
fi

if [ "${PREBUILT}" -eq 0 ]; then
    if [ -n "${COMPILER_PKG}" ]; then
        echo "installing ${BACKEND} compiler (${COMPILER_PKG})"
        # shellcheck disable=SC2086
        python3 -m pip install "${pip_args[@]}" ${COMPILER_PKG}
    fi
    if [ -n "${DEPS}" ]; then
        echo "installing ${BACKEND} deps (${DEPS# })"
        # Word splitting is intended: both are space-separated package lists.
        # shellcheck disable=SC2086
        python3 -m pip install "${pip_args[@]}" ${DEPS}
    fi
fi

# setuptools>=77 is the repo's build requirement (PEP 639 metadata); install it
# first so `--no-build-isolation` below has a backend to use.
python3 -m pip install "${pip_args[@]}" --upgrade "setuptools>=77"
python3 -m pip install "${pip_args[@]}" --no-build-isolation -e "${REPO_ROOT}"
python3 -m pip install "${pip_args[@]}" pytest pyyaml sqlalchemy packaging

echo "checking that FlagTrain imports on this platform"
python3 -c "import flag_train; print('flag_train:', flag_train.vendor_name, '/', flag_train.device)"
