"""Same sentence, two voices: proves session.update changes the voice itself.

Comparing two turns through the LLM is not evidence - the reply text differs,
so the audio would differ anyway. This drives the handler with one fixed
sentence and only the voice changes.

    python bench/probe_voice_style.py "Halo, apa kabar?" F2 M4

Run with SUPERTONIC_CACHE_DIR pointing at the q4 graphs, as run.ps1 does.
"""
import hashlib
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
        self.lang = "id"
        self.speed = 1.05
        self.blocksize = 512
        self.cancel_scope = None
        self.speculative_turns = None


def tts_input(text, voice):
    return NS(
        text=text,
        tts_language_code="id",
        selected_language="id",
        runtime_config=NS(session=NS(audio=NS(output=NS(voice=voice)))),
        response=None,
        turn_id="probe",
        turn_revision=0,
        response_key="probe",
    )


def render(handler, text, voice):
    """Mirror the handler's synthesize call and hash the PCM it produces."""
    style = handler._voice_style(handler._resolve_session_voice(tts_input(text, voice)))
    wav, _ = handler.tts.synthesize(text=text, lang="id", voice_style=style, speed=1.05)
    audio = np.asarray(wav, dtype=np.float32).squeeze()
    return hashlib.sha256(np.clip(audio, -1, 1).astype(np.float32).tobytes()).hexdigest()[:16], len(audio) / 44100


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Halo, apa kabar hari ini?"
    voices = sys.argv[2:] or ["F2", "M4"]

    handler = Probe("F1")
    used = {}
    for v in voices:
        digest, dur = render(handler, text, v)
        used[v] = digest
        print(f"{v}: {dur:5.2f}s audio  sha256:{digest}")

    digests = list(used.values())
    unique = len(set(digests))
    print(f"\n{len(digests)} voices -> {unique} distinct waveforms")
    if unique != len(digests):
        print("FAIL: at least two voices produced identical audio - session voice is being ignored")
        return 1
    print("OK: each requested voice produced different audio for the same sentence")

    # And the guard: an unknown name must fall back, not crash.
    fallback = handler._resolve_session_voice(tts_input(text, "zz"))
    print(f"unknown voice 'zz' -> falls back to {fallback!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
