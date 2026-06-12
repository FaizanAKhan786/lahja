#!/usr/bin/env bash
set -euo pipefail
export LAHJA_MODEL_DIR="${LAHJA_MODEL_DIR:-outputs/merged}"
uvicorn lahja.serve.api:app --host 0.0.0.0 --port "${PORT:-8000}"
