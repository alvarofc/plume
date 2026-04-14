#!/usr/bin/env bash
set -euo pipefail

# Download LateOn-Code-edge ONNX model from HuggingFace
# This is a 17M-param ColBERT model optimized for code retrieval

MODEL_DIR="${1:-models/lateon-code-edge}"
REPO="lightonai/LateOn-Code-edge"
HF_BASE="https://huggingface.co/${REPO}/resolve/main"

echo "Downloading LateOn-Code-edge to ${MODEL_DIR}..."
mkdir -p "${MODEL_DIR}"

# Required files
FILES=(
    "model.onnx"
    "tokenizer.json"
    "config.json"
    "special_tokens_map.json"
    "tokenizer_config.json"
)

for f in "${FILES[@]}"; do
    if [ -f "${MODEL_DIR}/${f}" ]; then
        echo "  [skip] ${f} already exists"
    else
        echo "  [download] ${f}"
        curl -fSL "${HF_BASE}/${f}" -o "${MODEL_DIR}/${f}"
    fi
done

echo ""
echo "Done. Model files are in ${MODEL_DIR}/"
echo ""
echo "To use with Plume, set in config.toml:"
echo "  [encoder]"
echo "  model = \"${MODEL_DIR}\""
echo ""
echo "And compile with: cargo build --release --features plume-encoder/onnx"
