#!/usr/bin/env bash
# Setup Ollama + qwen3.5:9b-mlx for Lyra on Apple Silicon (M3 / 18GB recommended).
set -euo pipefail

MODEL="${LYRA_OLLAMA_MODEL:-qwen3.5:9b-mlx}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

echo "==> Lyra Ollama setup"
echo "    Model: ${MODEL}"
echo "    Host:  ${HOST}"
echo
echo "Note: qwen3.5:9b-mlx is ~8.9GB. On 18GB unified memory, close heavy apps"
echo "      while the model is loaded. Lyra caps context at num_ctx=8192."
echo

if ! command -v ollama >/dev/null 2>&1; then
  echo "==> Ollama not found."
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This script targets macOS. Install Ollama from https://ollama.com/download"
    exit 1
  fi
  if command -v brew >/dev/null 2>&1; then
    echo "==> Installing Ollama via Homebrew..."
    brew install ollama
  else
    echo "Homebrew not found. Install Ollama from https://ollama.com/download then re-run."
    exit 1
  fi
fi

echo "==> Ensuring Ollama is reachable at ${HOST}..."
if ! curl -sf "${HOST}/api/tags" >/dev/null 2>&1; then
  echo "Ollama API is not responding."
  echo "Start it with: open -a Ollama   (or: ollama serve)"
  echo "Then re-run this script."
  exit 1
fi

echo "==> Pulling ${MODEL} (this may take a while)..."
ollama pull "${MODEL}"

echo "==> Verifying chat (think=false)..."
VERIFY_PAYLOAD=$(cat <<EOF
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
  "stream": false,
  "think": false,
  "options": {"num_ctx": 2048, "temperature": 0}
}
EOF
)

RESP=$(curl -sf "${HOST}/api/chat" -H "Content-Type: application/json" -d "${VERIFY_PAYLOAD}")
CONTENT=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("message") or {}).get("content") or d.get("response") or "")' <<<"${RESP}")

if [[ -z "${CONTENT}" ]]; then
  echo "Verification failed: empty response from Ollama."
  exit 1
fi

echo "    Model reply: ${CONTENT}"
echo
echo "==> Done. Start Lyra with:"
echo "    pip install -r requirements.txt"
echo "    python3 -m uvicorn lyra.server.app:app --host 0.0.0.0 --port 8000"
echo "    open http://localhost:8000"
