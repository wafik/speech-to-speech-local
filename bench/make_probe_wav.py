"""Render one Indonesian sentence with Supertonic q4 and save it as 16 kHz mono WAV.

Used as a hardware probe: if this file transcribes back word-correctly through
the GGUF STT server, both stages work on this machine.

    python bench/make_probe_wav.py out.wav "Halo, bisa bantu saya?"
"""
import os
import sys

import numpy as np
import soundfile as sf

# Same override run.ps1 uses; must be set before supertonic resolves its cache.
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("SUPERTONIC_CACHE_DIR", os.path.join(os.path.dirname(HERE), "models", "supertonic-3-q4"))

from supertonic import TTS  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "probe.wav"
text = sys.argv[2] if len(sys.argv) > 2 else "Halo, bisa bantu saya mengatur Docker container?"
voice = sys.argv[3] if len(sys.argv) > 3 else "F1"
lang = sys.argv[4] if len(sys.argv) > 4 else "id"

tts = TTS()
style = tts.get_voice_style(voice)
# synthesize() returns (wav, duration) in supertonic >=1.3
result = tts.synthesize(text, voice_style=style, lang=lang)
wav = result[0] if isinstance(result, tuple) else result
audio = np.asarray(wav, dtype=np.float32).squeeze()
sr = getattr(tts, "sample_rate", None) or 24000

# Pipeline STT expects 16 kHz mono.
if sr != 16000:
    count = int(round(len(audio) * 16000 / sr))
    audio = np.interp(np.linspace(0, len(audio) - 1, count), np.arange(len(audio)), audio).astype(np.float32)

sf.write(out, audio, 16000, subtype="PCM_16")
print(f"wrote {out}  {len(audio)/16000:.2f}s  (synth {sr} Hz, lang={lang}, voice={voice})")
print(f"expected text: {text}")
