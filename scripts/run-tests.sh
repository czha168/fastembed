#!/bin/bash
set -e
cd "$(dirname "$0")/.."
. .venv/bin/activate
python -m pytest tests/test_hardware.py tests/test_common.py -v
