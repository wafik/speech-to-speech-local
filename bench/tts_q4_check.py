"""Harder check on the q4 Supertonic variant before adopting it.

One sentence at 0% WER is not evidence. This runs several sentences, twice
each, and compares throughput against fp32 on the same machine state.
"""
import os
import re
import statistics
import time

import numpy as np
import torch
import transformers
from supertonic import TTS

HERE = os.path.dirname(os.path.abspath(__file__))
Q4 = os.path.join(os.path.dirname(HERE), "models", "supertonic-3-q4")
SENTS = [
    "Halo, saya siap membantu Anda hari ini. Silakan tanyakan apa saja.",
    "Tolong jelaskan cara melakukan deploy aplikasi ke server produksi.",
    "Apakah Anda bisa mengatur konfigurasi jaringan untuk kantor cabang?",
    "Saya ingin memesan tiket kereta api ke Surabaya besok pagi.",
    "Terima kasih sudah menghubungi kami, ada yang bisa saya bantu?",
]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


print("loading ASR...", flush=True)
proc = transformers.AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-0.6B-hf")
asr = transformers.AutoModelForMultimodalLM.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B-hf", dtype="float32"
).to("cuda").eval()


def transcribe(wav, sr):
    if sr != 16000:
        n = int(round(len(wav) * 16000 / sr))
        wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)
    inp = proc.apply_transcription_request(audio=wav).to("cuda", torch.float32)
    with torch.inference_mode():
        out = asr.generate(**inp, max_new_tokens=128)
    return proc.decode(out[:, inp["input_ids"].shape[1]:], return_format="transcription_only")[0].strip()


for label, d in (("fp32", None), ("q4", Q4)):
    kw = {"auto_download": True}
    if d:
        kw["model_dir"] = d
    tts = TTS(**kw)
    style = tts.get_voice_style(voice_name="F1")
    tts.synthesize("pemanasan", lang="id", voice_style=style)

    times, rtf, wers = [], [], []
    for s in SENTS:
        for _ in range(2):
            t0 = time.perf_counter()
            wav, _ = tts.synthesize(s, lang="id", voice_style=style)
            dt = time.perf_counter() - t0
            wav = np.asarray(wav, dtype=np.float32).squeeze()
            sr = int(tts.sample_rate)
            times.append(dt)
            rtf.append(dt / (len(wav) / sr))
            wers.append(wer(s, transcribe(wav, sr)))
    print(
        f"{label:5} median={statistics.median(times)*1000:6.0f}ms  "
        f"RTF={statistics.median(rtf):.3f}  "
        f"WER mean={np.mean(wers)*100:5.1f}%  max={np.max(wers)*100:5.1f}%  n={len(wers)}",
        flush=True,
    )
