# Lyra

Ambient-aware personal assistant ("Jarvis" style): continuous listening, speaker ID, rolling/episodic memory, and tap-to-talk answers via a **local Ollama** model.

## Hardware target

Optimized for **MacBook Pro M3 / 18GB unified memory** using:

| Setting | Value |
| --- | --- |
| Model | `qwen3.5:4b-mlx` (MLX, ~4.0GB, Text + Image; faster default) |
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
ollama pull qwen3.5:4b-mlx
```

## Run Lyra

```bash
docker compose up -d   # Qdrant for durable episodic memory
python3 -m pip install -r requirements.txt
python3 -m uvicorn lyra.server.app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**, enroll your voice with the **~60s scripted reading** in a natural conversational voice (ECAPA speaker embedding; first run downloads ONNX weights), then use **TAP TO TALK** / Spacebar.

### Ambient capture (mic + call/system audio)

Browser mic cannot hear Zoom/Meet output. For Wispr-like call awareness, run the native capture client (server must be up):

```bash
# List devices
python3 -m lyra.client.ambient_capture --list-devices

# Mic + BlackHole system/loopback mix (Mac)
python3 -m lyra.client.ambient_capture --system-device BlackHole
```

**Mac BlackHole setup**

1. Install [BlackHole 2ch](https://existential.audio/blackhole/).
2. Open **Audio MIDI Setup** → create a **Multi-Output Device** that includes your speakers **and** BlackHole.
3. Set macOS / call-app output to that Multi-Output Device (so you still hear audio while BlackHole receives a copy).
4. Point Lyra at BlackHole with `--system-device BlackHole` (or set `audio.system_device` in `config.json`).

With `asr.enabled: true` (default), the server runs **faster-whisper** on VAD utterances, tags Me/Not-Me with ECAPA, then optionally cleans text with **Ollama**. The Command Deck skips browser Web Speech for ambient when server ASR is on.

Re-enroll after speaker-matcher upgrades. Existing handcrafted 32-D `user_voice_profile.json` files are incompatible.

The Command Deck streams tokens from `POST /api/tap_to_talk/stream` (SSE) so the first words appear before generation finishes. Non-streaming `POST /api/tap_to_talk` remains available for CLI/compat.

If Ollama is not running, Lyra falls back to a heuristic synthesizer so the UI still responds.

## Configuration

See `config.json` → `agent.ollama`:

```json
"ollama": {
  "enabled": true,
  "host": "http://127.0.0.1:11434",
  "model": "qwen3.5:4b-mlx",
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
