#!/usr/bin/env bash
# Idempotent dependency setup for the Lyra Cloud Agent environment.
# Creates a project-local virtualenv and installs runtime + dev dependencies.
set -euo pipefail

cd "$(dirname "$0")/.."

# The default base image ships Python 3 but not always the venv/ensurepip
# module, which is required to create virtualenvs. Install it once if missing.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  echo "[install] python venv module missing; installing via apt..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq "python3-venv" || \
    sudo apt-get install -y -qq "python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')-venv"
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "[install] creating virtualenv at .venv"
  python3 -m venv .venv
fi

echo "[install] upgrading pip"
.venv/bin/python -m pip install --upgrade pip

echo "[install] installing runtime dependencies (requirements.txt)"
.venv/bin/python -m pip install -r requirements.txt

# Dev/optional dependencies not pinned in requirements.txt:
#  - pytest: required to run the test suite (tests/).
#  - beautifulsoup4: optional HTML fallback used by lyra/server/search.py.
echo "[install] installing dev/optional dependencies (pytest, beautifulsoup4)"
.venv/bin/python -m pip install pytest beautifulsoup4

echo "[install] done"
