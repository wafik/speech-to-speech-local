"""Integration smoke tests against the running speech-to-speech server.

  1. /v1/chat/completions  (LLM proxy — text agents can borrow the same model)
  2. /v1/realtime          (voice agent — send PCM, get PCM + transcript back)

Run:  python client_demo.py
"""
import asyncio, base64, json
import numpy as np
import websockets
from openai import OpenAI

BASE = "http://127.0.0.1:8765"


def text_agent():
    """Text-only agent: same LLM the voice pipeline uses, plain OpenAI SDK."""
    llm = OpenAI(base_url=f"{BASE}/v1", api_key="unused")
    r = llm.chat.completions.create(
        model="ignored",  # server forces its configured --model_name
        messages=[{"role": "user", "content": "Say hi in exactly five words."}],
        # the proxy forwards this verbatim; without it qwen3.5 thinks for 12k tokens
        extra_body={"reasoning_effort": "none"},
    )
    print("TEXT  :", r.choices[0].message.content.strip())


def speech_pcm(seconds=2.0):
    """Real speech, not a tone — VAD ignores tones. Needs `pip install kokoro soundfile`."""
    from kokoro import KPipeline
    import soundfile as sf
    import tempfile, os

    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    audio = np.concatenate([a for _, _, a in pipe("Hello, can you hear me clearly?", voice="af_heart")])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "u.wav")
        sf.write(path, audio, 24000)
        data, sr = sf.read(path, dtype="float32")
    n = int(len(data) * 16000 / sr)
    res = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data)
    return (np.clip(res, -1, 1) * 32767).astype("<i2").tobytes()


async def voice_agent():
    """Voice agent: stream 16 kHz mono PCM16 to /v1/realtime, collect the reply."""
    url = "ws://127.0.0.1:8765/v1/realtime"
    pcm = speech_pcm()
    transcript, audio = "", 0

    async with websockets.connect(url, max_size=None) as ws:
        async def pump():
            nonlocal transcript, audio
            async for raw in ws:
                ev = json.loads(raw)
                t = ev.get("type")
                if t == "response.output_audio.delta":
                    audio += len(base64.b64decode(ev["delta"]))
                elif t == "response.output_audio_transcript.delta":
                    transcript += ev["delta"]
                elif t == "response.done":
                    return

        task = asyncio.create_task(pump())
        for i in range(0, len(pcm), 640):  # 20 ms frames
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[i:i + 640]).decode(),
            }))
            await asyncio.sleep(0.02)
        await asyncio.sleep(1.0)  # let VAD close the turn
        try:
            await asyncio.wait_for(task, 90)
        except asyncio.TimeoutError:
            print("VOICE : timed out")
        finally:
            task.cancel()

    print("VOICE :", (transcript or "<no transcript>")[:120])
    # no session.update -> server's native rate, so decode at 16000
    print("AUDIO :", f"{audio / 2 / 16000:.1f}s of 16 kHz PCM")


if __name__ == "__main__":
    text_agent()
    asyncio.run(voice_agent())
