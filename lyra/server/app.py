import base64
import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lyra.server.agent import LyraAgentEngine, format_sse
from lyra.server.asr import WhisperASR
from lyra.server.enrollment_prompt import (
    MIN_COVERAGE_RATIO,
    MIN_DURATION_SEC,
    PROMPT_ID,
    coverage_ratio as compute_coverage_ratio,
    get_enrollment_prompt,
)
from lyra.server.ollama_client import OllamaClient
from lyra.server.rolling_memory import RollingMemoryEngine
from lyra.server.search import WebSearchEngine
from lyra.server.speaker_id import TargetSpeakerExtractor
from lyra.server.transcript_cleanup import TranscriptCleaner
from lyra.server.utterance_buffer import UtteranceBuffer
from lyra.server.vad import VoiceActivityDetector


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    threading.Thread(target=_warmup_models, name="lyra-warmup", daemon=True).start()
    yield


app = FastAPI(title="Lyra Ambient Assistant API", version="1.0.0", lifespan=_lifespan)
# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")
config = {
    "sample_rate": 16000,
    "vad_threshold": 0.015,
    "similarity_threshold": 0.28,
    "asr": {
        "enabled": True,
        "engine": "faster_whisper",
        "model": "small.en",
        "cleanup_enabled": True,
        "cleanup_timeout_seconds": 8.0,
        "device": "cpu",
        "compute_type": "int8",
    },
    "audio_devices": {
        "mic_device": None,
        "system_device": None,
        "mix_system": True,
        "system_gain": 1.0,
        "mic_gain": 1.0,
    },
    "agent": {
        "name": "Lyra",
        "persona": "Jarvis-style intelligent, ambient personal assistant. Concise, sharp, proactive, and contextually aware.",
        "web_search_enabled": False,
        "max_search_results": 4,
        "ollama": {
            "enabled": True,
            "host": "http://127.0.0.1:11434",
            "model": "qwen3.5:4b-mlx",
            "think": False,
            "num_ctx": 2048,
            "num_predict": 96,
            "temperature": 0.4,
            "keep_alive": -1,
            "timeout_seconds": 90,
        },
    },
}
memory_cfg = {
    "rolling_buffer_max_minutes": 30,
    "max_episodic_entries": 1000,
    "context_window_turns": 8,
}
qdrant_cfg = {
    "url": "http://localhost:6333",
    "collection": "lyra_episodic",
    "embedding_model": "google/embeddinggemma-300m",
    "sparse_model": "Qdrant/bm25",
    "vector_size": 768,
}

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg_data = json.load(f)
            config["sample_rate"] = cfg_data.get("audio", {}).get("sample_rate", 16000)
            config["vad_threshold"] = cfg_data.get("audio", {}).get("vad_energy_threshold", 0.015)
            config["similarity_threshold"] = cfg_data.get("audio", {}).get("speaker_similarity_threshold", 0.28)
            if isinstance(cfg_data.get("asr"), dict):
                config["asr"] = {**config["asr"], **cfg_data["asr"]}
            audio_block = cfg_data.get("audio", {})
            config["audio_devices"] = {
                "mic_device": audio_block.get("mic_device"),
                "system_device": audio_block.get("system_device"),
                "mix_system": bool(audio_block.get("mix_system", True)),
                "system_gain": float(audio_block.get("system_gain", 1.0)),
                "mic_gain": float(audio_block.get("mic_gain", 1.0)),
            }
            memory_cfg.update(cfg_data.get("memory", {}))
            nested_qdrant = memory_cfg.pop("qdrant", None) or cfg_data.get("memory", {}).get("qdrant")
            if nested_qdrant:
                qdrant_cfg.update(nested_qdrant)
            if isinstance(cfg_data.get("agent"), dict):
                config["agent"] = {**config["agent"], **cfg_data["agent"]}
                nested_ollama = cfg_data["agent"].get("ollama")
                if isinstance(nested_ollama, dict):
                    config["agent"]["ollama"] = {
                        **config["agent"].get("ollama", {}),
                        **nested_ollama,
                    }
    except Exception as e:
        print(f"[Server] Error reading config: {e}")

agent_cfg = config["agent"]
ollama_cfg = agent_cfg.get("ollama") if isinstance(agent_cfg.get("ollama"), dict) else {}
ollama_client = OllamaClient.from_config(ollama_cfg)

# Instantiate Core Engines
vad_detector = VoiceActivityDetector(sample_rate=config["sample_rate"], energy_threshold=config["vad_threshold"])
speaker_extractor = TargetSpeakerExtractor(profile_path="user_voice_profile.json", similarity_threshold=config["similarity_threshold"])
memory_engine = RollingMemoryEngine(
    max_buffer_minutes=int(memory_cfg.get("rolling_buffer_max_minutes", 30)),
    max_episodic_entries=int(memory_cfg.get("max_episodic_entries", 1000)),
    qdrant_url=str(qdrant_cfg.get("url", "http://localhost:6333")),
    qdrant_collection=str(qdrant_cfg.get("collection", "lyra_episodic")),
    embedding_model=str(qdrant_cfg.get("embedding_model", "google/embeddinggemma-300m")),
    sparse_model=str(qdrant_cfg.get("sparse_model", "Qdrant/bm25")),
    vector_size=int(qdrant_cfg.get("vector_size", 768)),
)
search_engine = WebSearchEngine(max_results=int(agent_cfg.get("max_search_results", 4)))
agent_engine = LyraAgentEngine(
    name=agent_cfg.get("name", "Lyra"),
    persona=agent_cfg.get("persona"),
    search_engine=search_engine,
    ollama_client=ollama_client,
    web_search_enabled=bool(agent_cfg.get("web_search_enabled", False)),
    context_window_turns=int(memory_cfg.get("context_window_turns", 8)),
)

asr_cfg = config.get("asr") if isinstance(config.get("asr"), dict) else {}
whisper_asr = WhisperASR(
    model_size=str(asr_cfg.get("model", "small.en")),
    enabled=bool(asr_cfg.get("enabled", True)),
    device=str(asr_cfg.get("device", "cpu")),
    compute_type=str(asr_cfg.get("compute_type", "int8")),
)
transcript_cleaner = TranscriptCleaner(
    ollama_client=ollama_client,
    enabled=bool(asr_cfg.get("cleanup_enabled", True)),
    timeout_seconds=float(asr_cfg.get("cleanup_timeout_seconds", 8.0)),
)

if ollama_client is None:
    print("[Server] Ollama disabled in config; using heuristic response synthesizer.")
elif ollama_client.is_reachable():
    model_ok = ollama_client.has_model()
    print(
        f"[Server] Ollama reachable at {ollama_client.host}; "
        f"model '{ollama_client.model}' {'found' if model_ok else 'NOT FOUND — run: ollama pull ' + ollama_client.model}."
    )
else:
    print(
        f"[Server] Ollama not reachable at {ollama_client.host}; "
        "tap-to-talk will fall back to heuristic responses until Ollama is running."
    )


def _warmup_models() -> None:
    """Load Ollama MLX weights + EmbeddingGemma so the first tap is warm."""
    print("[Server] Warming models in background...")
    try:
        if ollama_client is not None and ollama_client.is_reachable():
            ollama_client.warm()
            print(f"[Server] Ollama model '{ollama_client.model}' warmed (keep_alive={ollama_client.keep_alive}).")
        else:
            print("[Server] Skipping Ollama warm-up (unreachable or disabled).")
    except Exception as e:
        print(f"[Server] Ollama warm-up failed: {e}")

    try:
        store = getattr(memory_engine, "episodic_store", None)
        embedder = getattr(store, "embedder", None) if store is not None else None
        if embedder is not None and hasattr(embedder, "embed_query"):
            embedder.embed_query("warmup")
            print("[Server] Embedding models warmed.")
        else:
            print("[Server] Skipping embedder warm-up (no embedder available).")
    except Exception as e:
        print(f"[Server] Embedder warm-up failed: {e}")


class TapToTalkRequest(BaseModel):
    query: str
    force_search: bool = False


class VoiceEnrollRequest(BaseModel):
    user_name: str = "User"
    audio_base64: str
    heard_transcript: str | None = None
    coverage_ratio: float | None = None
    prompt_id: str | None = None


# Mount static files
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def get_dashboard():
    """Serves the Lyra Command Deck UI."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Lyra Ambient Personal Assistant</h1><p>Dashboard initializing...</p>")


@app.get("/api/status")
def get_system_status():
    """Returns server and engine status."""
    backend = memory_engine.backend_status()
    return {
        "status": "online",
        "timestamp": time.time(),
        "enrolled": speaker_extractor.enrolled_profile is not None,
        "enrolled_user": speaker_extractor.enrolled_metadata.get("user_name", "None"),
        "rolling_memory_entries": len(memory_engine.rolling_buffer),
        "episodic_memory_entries": memory_engine.episodic_count(),
        "episodic_backend": backend.get("episodic_backend", "qdrant"),
        "qdrant": backend.get("qdrant", {}),
        "asr": {
            **whisper_asr.status(),
            "cleanup": transcript_cleaner.status(),
        },
        "audio_devices": config.get("audio_devices", {}),
        "config": config,
    }


def _decode_ambient_audio(payload: dict) -> np.ndarray:
    """Decode float PCM from list floats or audio_base64 float32 bytes."""
    raw_audio = payload.get("audio", [])
    if isinstance(raw_audio, list) and raw_audio:
        return np.array(raw_audio, dtype=np.float32)

    b64 = payload.get("audio_base64")
    if isinstance(b64, str) and b64:
        try:
            audio_bytes = base64.b64decode(b64)
            return np.frombuffer(audio_bytes, dtype=np.float32).copy()
        except Exception:
            pass
    return np.zeros(0, dtype=np.float32)


def _process_server_utterance(
    utterance: np.ndarray,
    sample_rate: int,
    last_speech_speaker: dict,
) -> tuple[dict | None, dict]:
    """
    Run ECAPA + Whisper + cleanup on a completed utterance.
    Returns (transcript_entry, speaker_info).
    """
    speaker_info = speaker_extractor.identify_speaker(
        utterance,
        sample_rate=sample_rate,
        is_speech=True,
    )
    # Prefer committed utterance identity; fall back to sticky last speech.
    if not speaker_info.get("enrolled"):
        speaker_info = {**last_speech_speaker, **speaker_info}

    raw_text = whisper_asr.transcribe(utterance, sample_rate=sample_rate)
    if not raw_text.strip():
        return None, speaker_info

    cleaned = transcript_cleaner.clean(raw_text)
    entry = memory_engine.add_transcript(
        speaker=speaker_info.get("speaker_id", "User [Me]"),
        text=cleaned,
        confidence=float(speaker_info.get("confidence", 1.0)),
        is_user=bool(speaker_info.get("is_user", True)),
        is_final=True,
    )
    return entry, speaker_info


@app.websocket("/ws/ambient")
async def ambient_audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint for continuous background ambient audio streaming.
    Performs VAD, ECAPA speaker ID, optional server Whisper ASR + Ollama cleanup.
    """
    await websocket.accept()
    speaker_extractor.clear_stream_history()
    utterance_buffer = UtteranceBuffer(sample_rate=config["sample_rate"])
    last_speech_speaker = {
        "speaker_id": "User [Me]",
        "is_user": True,
        "confidence": 1.0,
    }
    server_asr_enabled = bool(whisper_asr.enabled)
    print(
        f"[WebSocket] Client connected for ambient streaming "
        f"(server_asr={'on' if server_asr_enabled else 'off'})."
    )

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            msg_type = payload.get("type", "audio_chunk")

            if msg_type in ("audio_chunk", "transcript_update"):
                text_transcript = payload.get("transcript", "")
                sample_rate = int(payload.get("sample_rate", config["sample_rate"]))
                audio_array = _decode_ambient_audio(payload)

                # Empty transcript-only frames: no VAD audio
                has_audio = audio_array.size > 0
                if not has_audio:
                    audio_array = np.zeros(1024, dtype=np.float32)

                vad_result = vad_detector.is_speech(audio_array) if has_audio else {
                    "is_speech": False,
                    "rms": 0.0,
                    "zcr": 0.0,
                    "confidence": 0.0,
                }

                if msg_type == "transcript_update" and not has_audio:
                    speaker_info = dict(last_speech_speaker)
                    speaker_info.setdefault("similarity_score", 0.0)
                    speaker_info.setdefault("enrolled", False)
                    speaker_info.setdefault("warmed", False)
                    speaker_info.setdefault("stable", False)
                else:
                    speaker_info = speaker_extractor.identify_speaker(
                        audio_array,
                        sample_rate=sample_rate,
                        is_speech=bool(vad_result.get("is_speech")),
                    )

                if vad_result.get("is_speech"):
                    last_speech_speaker = {
                        "speaker_id": speaker_info["speaker_id"],
                        "is_user": speaker_info["is_user"],
                        "confidence": speaker_info["confidence"],
                        "similarity_score": speaker_info.get("similarity_score", 0.0),
                        "enrolled": speaker_info.get("enrolled", False),
                        "warmed": speaker_info.get("warmed", False),
                        "stable": speaker_info.get("stable", False),
                    }

                transcript_entry = None

                # Server ASR path: buffer speech and transcribe on utterance end.
                if server_asr_enabled and has_audio:
                    completed = utterance_buffer.push(
                        audio_array, bool(vad_result.get("is_speech"))
                    )
                    if completed is not None and completed.size > 0:
                        loop = asyncio.get_event_loop()
                        completed_audio = completed
                        sr = sample_rate
                        sticky = dict(last_speech_speaker)

                        def _run_utt():
                            return _process_server_utterance(completed_audio, sr, sticky)

                        transcript_entry, utt_speaker = await loop.run_in_executor(None, _run_utt)
                        speaker_info = utt_speaker
                        last_speech_speaker = {
                            "speaker_id": utt_speaker.get("speaker_id", "User [Me]"),
                            "is_user": bool(utt_speaker.get("is_user", True)),
                            "confidence": utt_speaker.get("confidence", 1.0),
                            "similarity_score": utt_speaker.get("similarity_score", 0.0),
                            "enrolled": utt_speaker.get("enrolled", False),
                            "warmed": utt_speaker.get("warmed", True),
                            "stable": utt_speaker.get("stable", True),
                        }

                # Client Web Speech path only when server ASR is disabled.
                elif text_transcript.strip():
                    speaker_src = speaker_info if vad_result.get("is_speech") else last_speech_speaker
                    is_final = bool(payload.get("is_final", True))
                    transcript_entry = memory_engine.add_transcript(
                        speaker=speaker_src["speaker_id"],
                        text=text_transcript,
                        confidence=speaker_src.get("confidence", 1.0),
                        is_user=bool(speaker_src.get("is_user", True)),
                        is_final=is_final,
                    )

                await websocket.send_json({
                    "type": "stream_update",
                    "vad": vad_result,
                    "speaker": speaker_info,
                    "transcript_entry": transcript_entry,
                    "rolling_count": len(memory_engine.rolling_buffer),
                    "asr_enabled": server_asr_enabled,
                })

            elif msg_type == "tap_to_talk":
                query = payload.get("query", "")
                force_search = bool(payload.get("force_search", False))
                for event in agent_engine.process_tap_to_talk_stream(
                    query=query,
                    memory_engine=memory_engine,
                    force_search=force_search,
                ):
                    await websocket.send_json({"type": "tap_stream", "event": event})
                    if event.get("event") in ("done", "error"):
                        break

    except WebSocketDisconnect:
        speaker_extractor.clear_stream_history()
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        speaker_extractor.clear_stream_history()
        print(f"[WebSocket] Stream error: {e}")



@app.post("/api/tap_to_talk")
def tap_to_talk_handler(req: TapToTalkRequest):
    """
    HTTP trigger endpoint for user Tap-to-Talk query with ambient context synthesis.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = agent_engine.process_tap_to_talk(
        query=req.query,
        memory_engine=memory_engine,
        force_search=req.force_search,
    )
    return result


@app.post("/api/tap_to_talk/stream")
async def tap_to_talk_stream_handler(req: TapToTalkRequest):
    """SSE stream of tap-to-talk: status → token* → done|error.

    Uses an async generator that pulls sync Ollama/agent events via to_thread
    so each SSE frame is flushed to the client immediately (avoids buffering the
    whole reply until generation finishes).
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    stream_iter = agent_engine.process_tap_to_talk_stream(
        query=req.query,
        memory_engine=memory_engine,
        force_search=req.force_search,
    )
    sentinel = object()

    async def event_generator():
        while True:
            event = await asyncio.to_thread(next, stream_iter, sentinel)
            if event is sentinel:
                break
            yield format_sse(event)
            # Yield control so uvicorn can flush the chunk to the socket.
            await asyncio.sleep(0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/enroll_prompt")
def enroll_prompt():
    """Returns the predetermined voice enrollment reading script and thresholds."""
    return get_enrollment_prompt()


@app.post("/api/enroll_voice")
def enroll_voice_profile(req: VoiceEnrollRequest):
    """
    Enrolls user voice baseline profile from raw float audio array or base64 WAV.
    Requires a ~60s scripted reading when possible; rejects short or low-coverage takes.
    """
    try:
        if req.prompt_id is not None and req.prompt_id != PROMPT_ID:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown enrollment prompt_id '{req.prompt_id}'. Fetch /api/enroll_prompt.",
            )

        audio_bytes = base64.b64decode(req.audio_base64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        if len(audio_array) == 0:
            # Fallback mock array if testing with raw audio
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        duration_sec = float(len(audio_array) / config["sample_rate"]) if config["sample_rate"] else 0.0
        if duration_sec < MIN_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Enrollment audio too short ({duration_sec:.1f}s). "
                    f"Please read the full prompt for at least {MIN_DURATION_SEC}s."
                ),
            )

        resolved_coverage = req.coverage_ratio
        if req.heard_transcript:
            resolved_coverage = compute_coverage_ratio(req.heard_transcript)
            if resolved_coverage < MIN_COVERAGE_RATIO:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Enrollment coverage too low ({resolved_coverage:.0%}). "
                        f"Please read more of the prompt (need ≥ {MIN_COVERAGE_RATIO:.0%})."
                    ),
                )

        success = speaker_extractor.enroll_user(
            audio_array,
            user_name=req.user_name,
            sample_rate=config["sample_rate"],
            prompt_id=req.prompt_id or PROMPT_ID,
            coverage_ratio=resolved_coverage,
        )
        return {
            "success": success,
            "message": f"Successfully enrolled voice profile for '{req.user_name}'",
            "enrolled": True,
            "prototype_count": len(speaker_extractor.enrolled_prototypes),
            "coverage_ratio": resolved_coverage,
            "enrollment_duration_sec": round(duration_sec, 3),
            "model_id": speaker_extractor.MODEL_ID,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {e!s}")


@app.get("/api/memory")
def get_memory_log():
    """Returns stored rolling & episodic conversation memories."""
    return {
        "rolling_buffer": list(memory_engine.rolling_buffer),
        "episodic_memory": memory_engine.episodic_memory,
    }


@app.delete("/api/memory")
def clear_memory_log():
    """Clears stored memories."""
    memory_engine.clear_memory()
    return {"message": "Memory successfully cleared."}
