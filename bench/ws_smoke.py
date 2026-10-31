"""End-to-end smoke test: 16 kHz PCM WAV in -> transcript + reply audio out.

Run inside the WSL pipeline venv:
    .venv/bin/python /root/ws_smoke.py /root/test_id.wav

Prints the ASR transcript, the LLM reply text, and how many seconds of
16 kHz PCM came back, plus the wall-clock latency of one turn.
"""
import asyncio, base64, json, sys, time, wave

import websockets

URL = "ws://127.0.0.1:8765/v1/realtime"


def pcm16(path):
    with wave.open(path, "rb") as f:
        assert f.getframerate() == 16000, f.getframerate()
        assert f.getnchannels() == 1 and f.getsampwidth() == 2
        return f.readframes(f.getnframes())


async def main(path):
    pcm = pcm16(path)
    transcript, audio, events = "", 0, []
    t0 = time.time()
    first_audio = None

    async with websockets.connect(URL, max_size=None) as ws:
        async def pump():
            nonlocal transcript, audio, first_audio
            async for raw in ws:
                ev = json.loads(raw)
                t = ev.get("type")
                events.append(t)
                if t == "response.output_audio.delta":
                    if first_audio is None:
                        first_audio = time.time() - t0
                    audio += len(base64.b64decode(ev["delta"]))
                elif t in ("response.output_audio_transcript.delta",
                           "response.audio_transcript.delta"):
                    transcript += ev["delta"]
                elif t in ("response.output_audio_transcript.done",
                           "response.audio_transcript.done"):
                    transcript = ev.get("transcript", transcript) or transcript
                elif t == "response.done":
                    return

        task = asyncio.create_task(pump())
        for i in range(0, len(pcm), 640):          # 20 ms frames
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[i:i + 640]).decode(),
            }))
            await asyncio.sleep(0.02)
        await asyncio.sleep(1.0)                   # let VAD close the turn
        try:
            await asyncio.wait_for(task, 90)
        finally:
            task.cancel()

    print(f"IN    : {len(pcm) / 2 / 16000:.2f}s of 16 kHz PCM")
    print(f"REPLY : {transcript or '<none>'}")
    print(f"AUDIO : {audio / 2 / 16000:.2f}s of 16 kHz PCM")
    print(f"TTFA  : {first_audio:.2f}s" if first_audio else "TTFA  : <none>")
    print(f"TOTAL : {time.time() - t0:.2f}s")
    print(f"EVENTS: {len(events)} ({events[:3]} ... {events[-2:]})")


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "/root/test_id.wav"))
