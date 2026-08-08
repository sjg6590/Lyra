import asyncio
import json

import requests
import websockets

SERVER_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/ambient"

async def run_cli_streamer():
    """
    Headless CLI Streamer for Lyra ambient audio engine.
    Connects to home Linux server via WebSocket.
    """
    print("==================================================")
    print("      LYRA AMBIENT ASSISTANT — CLI STREAMER       ")
    print("==================================================")
    print(f"Connecting to server at: {WS_URL} ...")

    try:
        async with websockets.connect(WS_URL) as ws:
            print("✅ Connected to Lyra Ambient WebSocket Stream.")
            print("Type 'tap <query>' to invoke Jarvis, 'status' for server metrics, or 'exit' to quit.\n")

            while True:
                user_cmd = await asyncio.get_event_loop().run_in_executor(None, input, "Lyra-CLI > ")
                user_cmd = user_cmd.strip()

                if not user_cmd:
                    continue

                if user_cmd.lower() == "exit":
                    print("Exiting CLI streamer.")
                    break

                elif user_cmd.lower() == "status":
                    res = requests.get(f"{SERVER_URL}/api/status").json()
                    print(f"📊 Server Status: {json.dumps(res, indent=2)}")

                elif user_cmd.lower().startswith("tap"):
                    query = user_cmd[3:].strip() or "What was discussed in our recent conversation?"
                    print(f"✨ Sending Tap-to-Talk query: '{query}'...")

                    payload = {"query": query}
                    res = requests.post(f"{SERVER_URL}/api/tap_to_talk", json=payload).json()

                    print("\n---------------- LYRA RESPONSE ----------------")
                    print(f"🤖 Answer: {res['response']}")
                    print(f"⏱️ Latency: {res['latency_ms']} ms")
                    print("\n🧠 Thought Stream:")
                    for t in res.get("thoughts", []):
                        print(f"   ▸ {t}")

                    if res.get("search_results"):
                        print("\n🌐 Search Results Used:")
                        for s in res["search_results"]:
                            print(f"   • [{s['title']}]: {s['snippet']}")
                    print("-----------------------------------------------\n")

                else:
                    # Mock streaming text transcript entry
                    print(f"🎙️ Streaming mock ambient speech entry: '{user_cmd}'")
                    await ws.send(json.dumps({
                        "type": "audio_chunk",
                        "audio": [0.01] * 512,
                        "transcript": user_cmd,
                        "sample_rate": 16000
                    }))

                    # Receive response update
                    resp = await ws.receive()
                    msg = json.loads(resp)
                    print(f"📡 Server ACK: VAD={msg['vad']['is_speech']}, Speaker={msg['speaker']['speaker_id']}")

    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(run_cli_streamer())
