"""Prove that session.update actually changes the Supertonic voice.

Runs one realtime session against a local `serve`, sends a turn, switches the
voice with session.update, sends the same turn again, and compares the reply
audio. Different voice -> different waveform; same voice -> identical bytes up
to the LLM's own nondeterminism.

    python bench/probe_voice_switch.py bench/probe_id.wav F2 M4
"""
import asyncio
import base64
import hashlib
import json
import sys
import wave
from pathlib import Path

import websockets

URL = "ws://127.0.0.1:8765/v1/realtime"


def token() -> str:
    path = Path(__file__).resolve().parent.parent / "_ws_token"
    return path.read_text().strip() if path.is_file() else ""


def pcm16(path):
    with wave.open(path, "rb") as f:
        assert f.getframerate() == 16000 and f.getnchannels() == 1
        return f.readframes(f.getnframes())


async def one_turn(ws, pcm):
    audio = bytearray()
    transcript = ""
    done = asyncio.Event()

    async def pump():
        nonlocal transcript
        async for raw in ws:
            ev = json.loads(raw)
            t = ev.get("type")
            if t == "response.output_audio.delta":
                audio.extend(base64.b64decode(ev["delta"]))
            elif t in ("response.output_audio_transcript.delta", "response.audio_transcript.delta"):
                transcript += ev.get("delta", "")
            elif t == "response.done":
                done.set()
                return
            elif t == "error":
                raise RuntimeError(ev)

    task = asyncio.create_task(pump())
    for i in range(0, len(pcm), 640):
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm[i:i + 640]).decode(),
        }))
        await asyncio.sleep(0.02)
    await asyncio.wait_for(done.wait(), 120)
    task.cancel()
    return bytes(audio), transcript


async def set_voice(ws, voice):
    await ws.send(json.dumps({
        "type": "session.update",
        "event_id": f"probe_{voice}",
        "session": {"type": "realtime", "audio": {"output": {"voice": voice}}},
    }))
    await asyncio.sleep(0.5)  # let the server apply it before the next turn


async def main(path, first="F2", second="M4"):
    pcm = pcm16(path)
    tok = token()
    async with websockets.connect(URL, max_size=None, additional_headers=(
            {"Authorization": f"Bearer {tok}"} if tok else None)) as ws:
        await ws.recv()  # session.created
        results = {}
        for voice in (first, second):
            await set_voice(ws, voice)
            audio, text = await one_turn(ws, pcm)
            digest = hashlib.sha256(audio).hexdigest()[:16]
            results[voice] = digest
            print(f"voice {voice}: {len(audio)/2/16000:5.2f}s audio sha256:{digest}  {text[:70]!r}")
            await asyncio.sleep(1.0)  # let the slot release before the next turn

    a, b = results[first], results[second]
    print()
    print(f"{first} vs {second}: {'DIFFERENT audio' if a != b else 'IDENTICAL - voice did NOT change'}")
    return 0 if a != b else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(asyncio.run(main(args[0] if args else "bench/probe_id.wav",
                              args[1] if len(args) > 1 else "F2",
                              args[2] if len(args) > 2 else "M4")))
