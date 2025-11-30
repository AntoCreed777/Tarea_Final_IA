#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

echo "Initializing PDM project (non-interactive)..."
if ! command -v pdm >/dev/null 2>&1; then
	echo "Error: 'pdm' is not installed. Install it (e.g. pip install pdm-cli) and rerun this script." >&2
	exit 1
fi

# Try a non-interactive init when supported; fall back to interactive if it fails.
if pdm init -n >/dev/null 2>&1; then
	echo "pdm project initialized (non-interactive)."
else
	echo "pdm init -n not supported — running interactive 'pdm init'."
	pdm init
fi

echo "Adding development dependencies: black, isort, pydocstyle, pre-commit"
pdm add -d black isort pydocstyle pre-commit

echo "Installing pre-commit hooks (using project's environment)..."
pdm run pre-commit install

echo "Setup complete."