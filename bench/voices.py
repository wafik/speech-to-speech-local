"""Render the same Indonesian sentence in every Supertonic voice.

    python bench/voices.py            -> bench/audio_voice/*.wav
    python bench/voices.py "teks lain"
"""
import os
import sys
import wave

import numpy as np
from supertonic import TTS

TEXT = sys.argv[1] if len(sys.argv) > 1 else (
    "Halo, saya siap membantu Anda hari ini. Silakan tanyakan apa saja."
)
OUT = os.path.join(os.path.dirname(__file__), "audio_voice")
os.makedirs(OUT, exist_ok=True)

tts = TTS(auto_download=True)
sr = int(tts.sample_rate)
for name in ["F1", "F2", "F3", "F4", "F5", "M1"]:
    style = tts.get_voice_style(voice_name=name)
    wav, dur = tts.synthesize(text=TEXT, lang="id", voice_style=style)
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    path = os.path.join(OUT, f"{name}.wav")
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes())
    print(f"{name}  {len(wav) / sr:5.2f}s  {path}")
