#!/usr/bin/env bash
# Convert a merged HF model to quantized GGUF (Q4_K_M) for llama.cpp / Ollama.
# Usage: bash scripts/quantize_gguf.sh outputs/merged [Q4_K_M]
set -euo pipefail
MODEL_DIR="${1:?usage: quantize_gguf.sh <merged_model_dir> [quant_type]}"
QUANT="${2:-Q4_K_M}"

if [ ! -d "llama.cpp" ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp
  pip install -r llama.cpp/requirements.txt
  cmake -S llama.cpp -B llama.cpp/build && cmake --build llama.cpp/build -j --target llama-quantize
fi

python llama.cpp/convert_hf_to_gguf.py "$MODEL_DIR" --outfile outputs/model-f16.gguf
./llama.cpp/build/bin/llama-quantize outputs/model-f16.gguf "outputs/model-${QUANT,,}.gguf" "$QUANT"
echo "GGUF written: outputs/model-${QUANT,,}.gguf"
