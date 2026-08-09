"""
Native ambient capture client: mic + optional system/loopback mix → Lyra /ws/ambient.

Mac call capture setup (BlackHole):
  1. Install BlackHole 2ch (https://existential.audio/blackhole/)
  2. Audio MIDI Setup → create Multi-Output Device (Built-in Output + BlackHole)
  3. Set system/call output to the Multi-Output Device
  4. Run: python -m lyra.client.ambient_capture --system-device BlackHole
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import sys
import threading
import time
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None

try:
    import websocket
except ImportError:  # pragma: no cover - websockets package provides sync client optionally
    websocket = None


def list_input_devices() -> list[dict[str, Any]]:
    if sd is None:
        raise RuntimeError("sounddevice is not installed. pip install sounddevice")
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels", 0)) > 0:
            devices.append(
                {
                    "index": idx,
                    "name": dev.get("name", f"device-{idx}"),
                    "channels": int(dev.get("max_input_channels", 0)),
                    "default_samplerate": float(dev.get("default_samplerate", 16000)),
                }
            )
    return devices


def resolve_device(spec: str | int | None) -> int | None:
    if spec is None or spec == "":
        return None
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return int(spec)
    needle = str(spec).lower()
    for dev in list_input_devices():
        if needle in dev["name"].lower():
            return int(dev["index"])
    raise ValueError(f"No input device matching {spec!r}. Use --list-devices.")


def _to_mono(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return np.mean(arr, axis=1).astype(np.float32)


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32)
    duration = len(samples) / float(src_rate)
    target_len = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


class AmbientCaptureClient:
    def __init__(
        self,
        ws_url: str = "ws://127.0.0.1:8000/ws/ambient",
        sample_rate: int = 16000,
        block_size: int = 2048,
        mic_device: str | int | None = None,
        system_device: str | int | None = None,
        mix_system: bool = True,
        system_gain: float = 1.0,
        mic_gain: float = 1.0,
    ):
        if sd is None:
            raise RuntimeError("sounddevice is required: pip install sounddevice")
        self.ws_url = ws_url
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.mic_device = resolve_device(mic_device) if mic_device is not None else None
        self.system_device = None
        self.mix_system = bool(mix_system)
        self.system_gain = float(system_gain)
        self.mic_gain = float(mic_gain)
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._ws = None
        self._streams: list[Any] = []

        if self.mix_system and system_device:
            try:
                self.system_device = resolve_device(system_device)
            except ValueError as e:
                print(f"[Capture] WARNING: {e}; continuing mic-only.")
                self.system_device = None
        elif self.mix_system and not system_device:
            print("[Capture] mix_system enabled but no --system-device; mic-only.")

    def _open_stream(self, device: int | None, label: str):
        info = sd.query_devices(device)
        channels = 1 if int(info["max_input_channels"]) >= 1 else int(info["max_input_channels"])
        device_rate = int(info.get("default_samplerate") or self.sample_rate)

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                print(f"[Capture] {label} status: {status}")
            mono = _to_mono(indata)
            mono = _resample_linear(mono, device_rate, self.sample_rate)
            try:
                self._q.put_nowait((label, mono))
            except queue.Full:
                pass

        stream = sd.InputStream(
            device=device,
            channels=min(channels, 2),
            samplerate=device_rate,
            blocksize=max(256, int(self.block_size * device_rate / self.sample_rate)),
            dtype="float32",
            callback=callback,
        )
        stream.start()
        self._streams.append(stream)
        print(f"[Capture] Opened {label} device={device} ({info.get('name')}) @ {device_rate} Hz")

    def _connect_ws(self):
        # Prefer websocket-client if present; else use websockets sync bridge via thread.
        try:
            import websocket as ws_mod

            self._ws = ws_mod.create_connection(self.ws_url, timeout=10)
            print(f"[Capture] Connected to {self.ws_url}")
            return
        except Exception:
            pass

        # Fallback: websockets via a tiny sync wrapper
        import asyncio
        import websockets

        loop = asyncio.new_event_loop()

        class _AsyncWS:
            def __init__(self):
                self.conn = None

            def connect(self):
                self.conn = loop.run_until_complete(websockets.connect(self.ws_url))

            def send(self, data: str):
                loop.run_until_complete(self.conn.send(data))

            def close(self):
                if self.conn:
                    loop.run_until_complete(self.conn.close())

        wrapper = _AsyncWS()
        wrapper.connect()
        self._ws = wrapper
        print(f"[Capture] Connected to {self.ws_url} (websockets fallback)")

    def _send_audio(self, pcm: np.ndarray, source: str) -> None:
        if self._ws is None:
            return
        # Clip mix and encode as base64 float32 for compact transport.
        pcm = np.clip(pcm, -1.0, 1.0).astype(np.float32)
        payload = {
            "type": "audio_chunk",
            "audio_base64": base64.b64encode(pcm.tobytes()).decode("ascii"),
            "sample_rate": self.sample_rate,
            "source": source,
        }
        self._ws.send(json.dumps(payload))

    def run(self) -> None:
        self._connect_ws()
        self._open_stream(self.mic_device, "mic")
        if self.system_device is not None:
            self._open_stream(self.system_device, "system")

        mic_buf = np.zeros(0, dtype=np.float32)
        sys_buf = np.zeros(0, dtype=np.float32)
        target = self.block_size
        source_label = "mixed" if self.system_device is not None else "mic"

        print("[Capture] Streaming. Ctrl+C to stop.")
        try:
            while not self._stop.is_set():
                try:
                    label, chunk = self._q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if label == "mic":
                    mic_buf = np.concatenate([mic_buf, chunk * self.mic_gain])
                else:
                    sys_buf = np.concatenate([sys_buf, chunk * self.system_gain])

                while True:
                    if self.system_device is None:
                        if len(mic_buf) < target:
                            break
                        out = mic_buf[:target]
                        mic_buf = mic_buf[target:]
                    else:
                        if len(mic_buf) < target or len(sys_buf) < target:
                            break
                        out = mic_buf[:target] + sys_buf[:target]
                        mic_buf = mic_buf[target:]
                        sys_buf = sys_buf[target:]
                    self._send_audio(out, source_label)
        except KeyboardInterrupt:
            print("\n[Capture] Stopping...")
        finally:
            self._stop.set()
            for stream in self._streams:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            try:
                if self._ws is not None:
                    self._ws.close()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lyra ambient mic+system capture client")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8000/ws/ambient")
    parser.add_argument("--mic-device", default=None, help="Input device name substring or index")
    parser.add_argument("--system-device", default=None, help="Loopback device (e.g. BlackHole)")
    parser.add_argument("--no-mix-system", action="store_true", help="Mic only")
    parser.add_argument("--system-gain", type=float, default=1.0)
    parser.add_argument("--mic-gain", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args(argv)

    if args.list_devices:
        for dev in list_input_devices():
            print(f"[{dev['index']}] {dev['name']} (in={dev['channels']}, sr={dev['default_samplerate']})")
        return 0

    client = AmbientCaptureClient(
        ws_url=args.ws_url,
        sample_rate=args.sample_rate,
        mic_device=args.mic_device,
        system_device=None if args.no_mix_system else args.system_device,
        mix_system=not args.no_mix_system,
        system_gain=args.system_gain,
        mic_gain=args.mic_gain,
    )
    client.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
