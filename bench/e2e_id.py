"""End-to-end Indonesian voice test against the running speech-to-speech server.

Sends a 2.5 s Indonesian clip to /v1/realtime, prints the transcript and how
much audio came back. Run: python e2e_id.py [wav]
"""
import asyncio, base64, json, os, sys

import numpy as np
import soundfile as sf
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "ws://127.0.0.1:8765/v1/realtime"


def load(path):
    d, sr = sf.read(path, dtype="float32")
    if d.ndim > 1:
        d = d.mean(1)
    if sr != 16000:
        n = int(len(d) * 16000 / sr)
        d = np.interp(np.linspace(0, len(d) - 1, n), np.arange(len(d)), d)
    return (np.clip(d, -1, 1) * 32767).astype("<i2").tobytes()


async def main(path):
    pcm = load(path)
    transcript, audio = "", 0
    async with websockets.connect(URL, max_size=None) as ws:
        async def pump():
            nonlocal transcript, audio
            async for raw in ws:
                ev = json.loads(raw)
                t = ev.get("type")
                if t == "response.output_audio.delta":
                    audio += len(base64.b64decode(ev["delta"]))
                elif t == "response.output_audio_transcript.delta":
                    transcript += ev["delta"]
                elif t == "error":
                    print("ERR :", ev)
                    return
                elif t == "response.done":
                    return

        task = asyncio.create_task(pump())
        for i in range(0, len(pcm), 640):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[i:i + 640]).decode(),
            }))
            await asyncio.sleep(0.02)
        await asyncio.sleep(1.2)
        try:
            await asyncio.wait_for(task, 90)
        except asyncio.TimeoutError:
            print("VOICE: timed out")
        finally:
            task.cancel()

    print("USER :", f"{len(pcm)/2/16000:.2f}s in")
    print("BOT  :", (transcript or "<no transcript>")[:200])
    print("AUDIO:", f"{audio/2/16000:.2f}s out @16kHz")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "audio_short.wav")))
