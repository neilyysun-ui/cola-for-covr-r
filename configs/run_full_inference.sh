#!/usr/bin/env bash
set -euo pipefail

: "${TEST_JSON:?set TEST_JSON}"
: "${VIDEO_DIR:?set VIDEO_DIR}"
: "${OUTPUT_JSON:?set OUTPUT_JSON}"

args=(
  --test-json "${TEST_JSON}"
  --video-dir "${VIDEO_DIR}"
  --output-json "${OUTPUT_JSON}"
)

if [[ -n "${CANDIDATE_POOL:-}" ]]; then
  args+=(--candidate-pool "${CANDIDATE_POOL}")
fi

if [[ -n "${GEMINI_KEY_FILE:-}" ]]; then
  args+=(--gemini-key "${GEMINI_KEY_FILE}")
fi

if [[ -n "${MODEL:-}" ]]; then
  args+=(--model "${MODEL}")
fi

if [[ -n "${DEVICE:-}" ]]; then
  args+=(--device "${DEVICE}")
fi

python run_final.py "${args[@]}"
