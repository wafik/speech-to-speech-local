"""Same sentence, two voices: proves session.update changes the voice itself.

Comparing two turns through the LLM is not evidence - the reply text differs,
so the audio would differ anyway. This drives the handler with one fixed
sentence and only the voice changes.

Supertonic's vocoder is stochastic, so the PCM is never byte-identical even for
the same voice twice. The comparison is therefore spectral: a short-time
magnitude spectrum per clip, then the distance between clips. Two renders of one
voice sit far closer together than renders of two different voices.

    python bench/probe_voice_style.py "Halo, apa kabar?" F2 M4

Run with SUPERTONIC_CACHE_DIR pointing at the q4 graphs, as run.ps1 does.
"""
import sys
from types import SimpleNamespace as NS

import numpy as np

from speech_to_speech.TTS.supertonic_tts_handler import SupertonicTTSHandler


class Probe(SupertonicTTSHandler):
    """Skips the pipeline base class; only the pieces the voice path touches."""

    def __init__(self, voice="F1"):
        from supertonic import TTS

        self.tts = TTS(auto_download=True)
        self.voice = voice
        self.voice_style = self.tts.get_voice_style(voice_name=voice)
        self._style_cache = {voice: self.voice_style}
        self._response_voice = {}
        self.lang = "id"
        self.speed = 1.05
        self.blocksize = 512
        self.cancel_scope = None
        self.speculative_turns = None


def tts_input(text, voice, key=None):
    return NS(
        text=text,
        tts_language_code="id",
        selected_language="id",
        runtime_config=NS(session=NS(audio=NS(output=NS(voice=voice)))),
        response=None,
        turn_id="probe",
        turn_revision=0,
        response_key=key,
    )


def low_band_ratio(audio, lo=80, hi=200, sr=44100, n_fft=2048):
    """Share of the speech energy in the 80-200 Hz band: the pitch/timbre proxy.

    A male voice puts several times more energy there than a female one, and it
    is stable across renders of the same voice. That is the cheapest measurable
    difference, which is all this probe needs - it is not a voice ID.
    """
    audio = np.asarray(audio, dtype=np.float32).squeeze()
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


def render(handler, text, voice, key=None):
    """Mirror the handler's synthesize call; return (audio, voice actually used)."""
    tts_in = tts_input(text, voice, key)
    used = handler._response_voice_for(tts_in)
    wav, _ = handler.tts.synthesize(text=text, lang="id", voice_style=handler._voice_style(used), speed=1.05)
    return np.asarray(wav, dtype=np.float32).squeeze(), used


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Halo, apa kabar hari ini?"
    voices = sys.argv[2:] or ["F2", "M4"]

    handler = Probe("F1")
    clips = {}
    for v in voices:
        clips[v] = [render(handler, text, v) for _ in range(3)]
        used = clips[v][0][1]
        print(f"{v}: {len(clips[v][0][0]) / 44100:5.2f}s audio, voice used {used!r}")

    # Noise floor: the spread of the metric across renders of ONE voice.
    ok = True
    spread = 0.0
    for v, renders in clips.items():
        values = [low_band_ratio(c) for c, _ in renders]
        spread = max(spread, max(values) - min(values))
        print(f"  {v}: low-band ratio {np.mean(values):.3f}  (spread {max(values) - min(values):.3f})")
    print(f"same-voice spread: {spread:.3f}")

    if len(voices) > 1:
        means = {v: float(np.mean([low_band_ratio(c) for c, _ in clips[v]])) for v in voices}
        gap = abs(means[voices[1]] - means[voices[0]])
        print(f"\n{voices[0]} vs {voices[1]}: {gap:.3f} apart  ({gap / max(spread, 1e-6):.1f}x the within-voice spread)")
        print(f"  {voices[0]}={means[voices[0]]:.3f}  {voices[1]}={means[voices[1]]:.3f}")
        if gap <= spread * 2:
            print("FAIL: two voices are acoustically indistinguishable - session voice is ignored")
            ok = False
        else:
            print("OK: the requested voices are acoustically distinct")

    # Guard: an unknown name must fall back, not crash.
    fallback = handler._resolve_session_voice(tts_input(text, "zz"))
    print(f"unknown voice 'zz' -> falls back to {fallback!r}")

    # Mid-response switch must not splice voices: one response_key, the first
    # sentence decides. A second sentence asking for M4 must stay on F2.
    first, used_first = render(handler, text, "F2", key="resp_frozen")
    second, used_second = render(handler, text, "M4", key="resp_frozen")
    print(f"\nresponse_key frozen: first={used_first!r} second={used_second!r}")
    if used_second != used_first:
        print("FAIL: a mid-response voice switch reached the second sentence")
        ok = False
    else:
        print("OK: the mid-response switch was held until the response ended")

    # After the response ends the switch applies to the next one.
    handler._response_voice.pop("resp_frozen", None)
    _, used_next = render(handler, text, "M4", key="resp_next")
    print(f"next response: voice {used_next!r}")
    if used_next != "M4":
        print("FAIL: the switch did not take effect on the next response")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
