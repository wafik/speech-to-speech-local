"""A voice switch sent mid-reply must not splice two voices into one answer.

Regression for the freeze-per-response rule. Runs one realtime session against a
local `serve`, sends a turn, waits until the reply has started, then fires a
session.update for a different voice while audio is still streaming, and finally
checks that the whole reply is one voice while the NEXT reply got the new one.

    python bench/probe_voice_freeze.py bench/probe_id.wav F2 M4
"""
import asyncio
import base64
import json
import sys
import wave
from pathlib import Path

import numpy as np
import websockets

URL = "ws://127.0.0.1:8765/v1/realtime"


def token() -> str:
    path = Path(__file__).resolve().parent.parent / "_ws_token"
    return path.read_text().strip() if path.is_file() else ""


def pcm16(path):
    with wave.open(path, "rb") as f:
        assert f.getframerate() == 16000 and f.getnchannels() == 1
        return f.readframes(f.getnframes())


def low_band_ratio(audio, lo=80, hi=200, sr=16000, n_fft=1024):
    """Male voices carry several times more energy at 80-200 Hz; see probe_voice_style."""
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (freqs >= lo) & (freqs < hi)
    window = np.hanning(n_fft)
    ratios = []
    for start in range(0, audio.size - n_fft + 1, n_fft // 2):
        mag = np.abs(np.fft.rfft(audio[start:start + n_fft] * window))
        total = mag.sum()
        if total > 0:
            ratios.append(mag[band].sum() / total)
    return float(np.mean(ratios)) if ratios else 0.0


async def set_voice(ws, voice):
    await ws.send(json.dumps({
        "type": "session.update",
        "event_id": f"probe_{voice}_{np.random.default_rng().integers(1e6)}",
        "session": {"type": "realtime", "audio": {"output": {"voice": voice}}},
    }))


async def one_turn(ws, pcm, switch_at=None, switch_to=None):
    """Stream one turn; optionally send a session.update once audio starts."""
    audio = bytearray()
    first_delta = asyncio.Event()
    switched = False
    done = asyncio.Event()
    done_evt = asyncio.get_running_loop().create_future()

    async def pump():
        nonlocal switched
        async for raw in ws:
            ev = json.loads(raw)
            t = ev.get("type")
            if t == "response.output_audio.delta":
                audio.extend(base64.b64decode(ev["delta"]))
                if not switched:
                    first_delta.set()
            elif t == "response.done":
                done_evt.set_result(True)
                return
            elif t == "error":
                print("server error:", ev)
                done_evt.set_result(False)
                return

    task = asyncio.create_task(pump())
    for i in range(0, len(pcm), 640):
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm[i:i + 640]).decode(),
        }))
        await asyncio.sleep(0.02)

    if switch_at is not None:
        # Fire the switch while the reply is still being spoken.
        try:
            await asyncio.wait_for(first_delta.wait(), timeout=90)
            await asyncio.sleep(0.35)
            await set_voice(ws, switch_to)
            switched = True
            print(f"  (voice switch to {switch_to} sent mid-reply)")
        except asyncio.TimeoutError:
            print("  (no audio started; switch not sent)")

    await asyncio.wait_for(done_evt, timeout=180)
    task.cancel()
    pcm_out = np.frombuffer(bytes(audio), dtype=np.int16).astype(np.float32) / 32768.0
    return pcm_out


async def main(path, first="F2", second="M4"):
    pcm = pcm16(path)
    tok = token()
    async with websockets.connect(URL, max_size=None, additional_headers=(
            {"Authorization": f"Bearer {tok}"} if tok else None)) as ws:
        await ws.recv()  # session.created
        await set_voice(ws, first)
        await asyncio.sleep(0.5)

        print(f"turn 1: voice {first}, switch to {second} mid-reply")
        reply_a = await one_turn(ws, pcm, switch_at=True, switch_to=second)
        await asyncio.sleep(1.0)

        print(f"turn 2: should be {second} from the start")
        reply_b = await one_turn(ws, pcm)
        await asyncio.sleep(1.0)

    if reply_a.size == 0 or reply_b.size == 0:
        print("FAIL: no audio captured")
        return 1

    # Split each reply in half: a spliced voice shows up as one half matching
    # the other voice. Compare halves against the two full replies.
    def halves(x):
        mid = x.size // 2
        return x[:mid], x[mid:]

    a1, a2 = halves(reply_a)
    b1, _ = halves(reply_b)
    r = {k: round(low_band_ratio(v), 3) for k, v in
         {"reply1_first_half": a1, "reply1_second_half": a2, "reply2": b1}.items()}
    print("\nlow-band ratio (80-200 Hz):", r)

    ok = True
    # Within one reply the halves must be the same voice: a switch that leaked
    # through would make the second half jump toward reply2's value.
    lag = abs(r["reply1_second_half"] - r["reply1_first_half"])
    gap = abs(r["reply2"] - r["reply1_first_half"])
    print(f"\nwithin reply 1: halves differ by {lag:.3f}")
    print(f"reply 1 vs reply 2: {gap:.3f} apart")
    if gap < 0.02:
        print("FAIL: the mid-reply switch never took effect on the next reply")
        ok = False
    if lag > gap * 0.5:
        print("FAIL: reply 1 changed voice halfway - the freeze did not hold")
        ok = False
    if ok:
        print("OK: reply 1 kept one voice despite the mid-reply switch; reply 2 used the new one")
    return 0 if ok else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(asyncio.run(main(
        args[0] if args else "bench/probe_id.wav",
        args[1] if len(args) > 1 else "F2",
        args[2] if len(args) > 2 else "M4",
    )))
