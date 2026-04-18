#!/usr/bin/env bash
set -euo pipefail

# Download BEIR SciFact dataset for recall benchmarking
# 5,183 scientific documents, 300 test queries with relevance judgments

DATA_DIR="${1:-data/scifact}"

echo "Downloading BEIR SciFact to ${DATA_DIR}..."
mkdir -p "${DATA_DIR}"

if [ -f "${DATA_DIR}/corpus.jsonl" ] && [ -f "${DATA_DIR}/queries.jsonl" ] && [ -d "${DATA_DIR}/qrels" ]; then
    echo "  [skip] SciFact already downloaded"
else
    TMP_ZIP="/tmp/scifact-beir.zip"
    curl -fSL "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip" -o "${TMP_ZIP}"
    unzip -o "${TMP_ZIP}" -d /tmp/scifact-extract
    # The zip contains a scifact/ subdirectory
    cp -r /tmp/scifact-extract/scifact/* "${DATA_DIR}/"
    rm -rf /tmp/scifact-extract "${TMP_ZIP}"
fi

echo ""
echo "Done. Dataset files:"
echo "  ${DATA_DIR}/corpus.jsonl    ($(wc -l < "${DATA_DIR}/corpus.jsonl") docs)"
echo "  ${DATA_DIR}/queries.jsonl   ($(wc -l < "${DATA_DIR}/queries.jsonl") queries)"
echo "  ${DATA_DIR}/qrels/test.tsv  ($(wc -l < "${DATA_DIR}/qrels/test.tsv") lines)"
echo ""
echo "Run the recall benchmark:"
echo "  PROTOC=\$(which protoc) cargo run --release -p plume-bench --bin bench-recall --features plume-encoder/onnx"
