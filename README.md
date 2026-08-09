# Lyra

Ambient-aware personal assistant ("Jarvis" style): continuous listening, speaker ID, rolling/episodic memory, and tap-to-talk answers via a **local Ollama** model.

## Hardware target

Optimized for **MacBook Pro M3 / 18GB unified memory** using:

| Setting | Value |
| --- | --- |
| Model | `qwen3.5:9b-mlx` (MLX, ~8.9GB, Text + Image) |
| Context | `num_ctx: 2048` (keep prefill small for latency) |
| Max tokens | `num_predict: 96` (short spoken replies) |
| Keep-alive | `-1` (keep model resident after warm-up) |
| Thinking | disabled (`think: false`) for spoken, low-latency replies |

Close heavy apps while the model is loaded so macOS + Lyra + the browser UI still fit. Cold first request after reboot can still be slow; Lyra warms Ollama + embeddings on startup for subsequent taps.

## Prerequisites (Mac)

1. Install [Ollama](https://ollama.com/download) (or Homebrew: `brew install ollama`).
2. Start Ollama (`open -a Ollama` or `ollama serve`).
3. Pull the model and verify:

```bash
chmod +x scripts/setup_ollama_mac.sh
./scripts/setup_ollama_mac.sh
```

Or manually:

```bash
ollama pull qwen3.5:9b-mlx
```

## Run Lyra

```bash
docker compose up -d   # Qdrant for durable episodic memory
python3 -m pip install -r requirements.txt
python3 -m uvicorn lyra.server.app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**, start ambient listening, enroll your voice, then use **TAP TO TALK** / Spacebar.

The Command Deck streams tokens from `POST /api/tap_to_talk/stream` (SSE) so the first words appear before generation finishes. Non-streaming `POST /api/tap_to_talk` remains available for CLI/compat.

If Ollama is not running, Lyra falls back to a heuristic synthesizer so the UI still responds.

## Configuration

See `config.json` → `agent.ollama`:

```json
"ollama": {
  "enabled": true,
  "host": "http://127.0.0.1:11434",
  "model": "qwen3.5:9b-mlx",
  "think": false,
  "num_ctx": 2048,
  "num_predict": 96,
  "temperature": 0.4,
  "keep_alive": -1,
  "timeout_seconds": 90
}
```

Latency-related agent settings:

- `agent.web_search_enabled` — must be `true` (or `force_search`) before DuckDuckGo runs on the critical path
- `memory.context_window_turns` — ambient turns injected into the prompt (default `8`)

More architecture detail: [architecture_guide.md](architecture_guide.md).
