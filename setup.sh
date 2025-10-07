#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
cd "$SCRIPT_DIR"

pip install -e .
