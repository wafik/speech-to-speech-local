"""Render one sentence in fp32 and q4 Supertonic, for a side-by-side listen.

    python bench/tts_listen.py              # default sentence, voice F1
    python bench/tts_listen.py "teks lain" F3
"""
import os
import sys
import wave

import numpy as np
from supertonic import TTS

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = sys.argv[1] if len(sys.argv) > 1 else (
    "Halo, saya siap membantu Anda hari ini. Silakan tanyakan apa saja."
)
VOICE = sys.argv[2] if len(sys.argv) > 2 else "F1"
OUT = os.path.join(HERE, "audio_voice")
os.makedirs(OUT, exist_ok=True)

for label, d in (("fp32", None), ("q4", os.path.join(os.path.dirname(HERE), "models", "supertonic-3-q4"))):
    kw = {"auto_download": True}
    if d:
        kw["model_dir"] = d
    tts = TTS(**kw)
    style = tts.get_voice_style(voice_name=VOICE)
    wav, _ = tts.synthesize(TEXT, lang="id", voice_style=style)
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    sr = int(tts.sample_rate)
    path = os.path.join(OUT, f"{label}_{VOICE}.wav")
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes())
    print(f"{label:5} {len(wav)/sr:5.2f}s  {path}")
