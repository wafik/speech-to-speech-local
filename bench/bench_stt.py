"""Latency benchmark: Qwen3-ASR-0.6B vs faster-whisper medium-id, on Indonesian FLEURS clips.

Mirrors exactly what speech_to_speech/STT/*_handler.py does, without the pipeline.
Usage: python bench_stt.py [qwen|fw|both]
"""
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FW_ID = "cahya/faster-whisper-medium-id"
RESULT = os.path.join(HERE, "bench_result.json")


def load_clips():
    clips = []
    for p in sorted(glob.glob(os.path.join(HERE, "audio", "*.wav"))):
        data, sr = sf.read(p, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:  # pipeline is 16 kHz end to end
            n = int(len(data) * 16000 / sr)
            data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data)
        clips.append((os.path.basename(p)[:-4], data.astype(np.float32)))
    return clips


def bench_qwen(clips):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    t0 = time.perf_counter()
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-0.6B-hf")
    model = AutoModelForMultimodalLM.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B-hf", dtype=torch.bfloat16
    ).to("cuda").eval()
    load_s = time.perf_counter() - t0

    def run(audio, language="id"):
        req = {"audio": audio}
        if language:
            req["language"] = language
        inputs = proc.apply_transcription_request(**req).to("cuda", torch.bfloat16)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=256)
        gen = ids[:, inputs["input_ids"].shape[1]:]
        return proc.decode(gen, return_format="transcription_only")[0].strip()

    run(clips[0][1])  # warmup
    lat, out = [], []
    for cid, audio in clips:
        t = time.perf_counter()
        text = run(audio)
        lat.append(time.perf_counter() - t)
        out.append((cid, text))
    vram = torch.cuda.max_memory_allocated() / 1e9
    return load_s, vram, lat, out


def bench_fw(clips):
    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel(FW_ID, device="cuda", compute_type="int8_float16")
    load_s = time.perf_counter() - t0

    def run(audio):
        segs, _ = model.transcribe(audio, language="id", beam_size=1, without_timestamps=True)
        return " ".join(s.text for s in segs).strip()

    run(clips[0][1])  # warmup
    lat, out = [], []
    for cid, audio in clips:
        t = time.perf_counter()
        text = run(audio)
        lat.append(time.perf_counter() - t)
        out.append((cid, text))
    vram = torch.cuda.max_memory_allocated() / 1e9
    return load_s, vram, lat, out


if __name__ == "__main__":
    which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    clips = load_clips()
    dur = [len(a) / 16000 for _, a in clips]
    print(f"clips: {len(clips)} | audio {min(dur):.2f}-{max(dur):.2f}s", flush=True)

    results = {}
    if os.path.exists(RESULT):
        results = json.load(open(RESULT, encoding="utf-8"))

    for name, fn, key in (
        ("qwen3-asr-0.6B", bench_qwen, "qwen"),
        ("faster-whisper medium-id", bench_fw, "fw"),
    ):
        if which not in (key, "both"):
            continue
        torch.cuda.reset_peak_memory_stats()
        load_s, vram, lat, out = fn(clips)
        torch.cuda.empty_cache()
        med = sorted(lat)[len(lat) // 2]
        summary = {
            "load_s": round(load_s, 1),
            "vram_gb": round(vram, 2),
            "median_ms": round(med * 1000),
            "min_ms": round(min(lat) * 1000),
            "max_ms": round(max(lat) * 1000),
            "per_s_audio_ms": round(med / np.mean(dur) * 1000),
            "texts": {cid: text for cid, text in out},
        }
        results[key] = summary
        print(f"\n== {name} ==", flush=True)
        print(f"  load {load_s:.1f}s | peak VRAM {vram:.2f} GB | median {med*1000:.0f} ms "
              f"| range {min(lat)*1000:.0f}-{max(lat)*1000:.0f} ms", flush=True)
        for cid, text in out:
            print(f"  [{cid}] {text}", flush=True)

    json.dump(results, open(RESULT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nsaved ->", RESULT, flush=True)
