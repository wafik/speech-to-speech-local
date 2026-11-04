"""Compare transcribe.cpp backends on this machine: load time + per-clip latency.

    python bench/probe_backend.py cpu vulkan
"""
import sys
import time

import numpy as np
import soundfile as sf
from transcribe_cpp import Model

model_path = "models/qwen-asr-gguf/Qwen3-ASR-0.6B-Q5_K_M.gguf"
audio, sr = sf.read("bench/probe_id.wav", dtype="float32", always_2d=True)
pcm = np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32)
print(f"clip: {len(pcm)/sr:.2f}s @ {sr} Hz")

for backend in (sys.argv[1:] or ["cpu", "vulkan"]):
    t0 = time.perf_counter()
    model = Model(model_path, backend=backend)
    load = time.perf_counter() - t0
    session = model.session()
    try:
        session.run(pcm, language=None, timestamps="none")  # warm
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            result = session.run(pcm, language=None, timestamps="none")
            times.append(time.perf_counter() - t0)
        best = min(times)
        print(
            f"{backend:8s} backend={model.backend:10s} load={load:5.2f}s "
            f"latency={best*1000:6.0f}ms RTF={best/(len(pcm)/sr):.2f}  {result.text}"
        )
    finally:
        session.close()
        model.close()
