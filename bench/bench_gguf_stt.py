"""Compare quantized Qwen and Whisper GGUFs on the existing Indonesian FLEURS clips."""
import glob
import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
from transcribe_cpp import Model

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF = {
    "1515": "ini adalah cep kelima martelly dalam empat tahun",
    "1531": "kepulauan afrika timur berada di samudra hindia di lepas pantai timur afrika",
    "1545": "kari bisa menjadi kering atau basah tergantung dari jumlah cairannya",
    "1555": "di pelataran gereja ada pahatan marmer merpati menarik di atas sejumlah makam",
    "1560": "begitu keluar dari arus berenang kembali tidak sesulit biasanya",
    "1622": "presiden as george w bush menerima pengumuman tersebut",
    "1648": "tidak ada hutan besar di tanah kanaan itu sebabnya harga kayu sangat mahal",
    "1656": "polisi mengatakan bahwa mayat itu sudah satu hari berada di tempat tersebut",
}
MODELS = [
    ("qwen-q8", "models/qwen-asr-gguf/Qwen3-ASR-0.6B-Q8_0.gguf"),
    ("qwen-q6", "models/qwen-asr-gguf/Qwen3-ASR-0.6B-Q6_K.gguf"),
    ("qwen-q5", "models/qwen-asr-gguf/Qwen3-ASR-0.6B-Q5_K_M.gguf"),
    ("qwen-q4", "models/qwen-asr-gguf/Qwen3-ASR-0.6B-Q4_K_M.gguf"),
    ("whisper-q8", "models/whisper-turbo-gguf/whisper-large-v3-turbo-Q8_0.gguf"),
    ("whisper-q4", "models/whisper-turbo-gguf/whisper-large-v3-turbo-Q4_K_M.gguf"),
]


def norm(text):
    return re.sub(r"[^\w\s]|_", " ", text.lower().replace("-", " ")).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[-1] / max(1, len(r))


def gpu_memory_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.CalledProcessError, IndexError):
        return None


def load_clips():
    clips = []
    for path in sorted(glob.glob(os.path.join(HERE, "audio", "*.wav"))):
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            n = int(round(len(audio) * 16000 / sr))
            audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
        clips.append((os.path.basename(path)[:-4], np.asarray(audio, dtype=np.float32)))
    if len(clips) != len(REF):
        raise RuntimeError(f"expected {len(REF)} clips, found {len(clips)}")
    return clips


def benchmark(label, relpath, clips):
    path = os.path.join(ROOT, relpath)
    baseline_mb = gpu_memory_mb()
    load_t = time.perf_counter()
    texts, latencies = {}, []
    with Model(path, backend="vulkan") as model:
        with model.session() as session:
            load_s = time.perf_counter() - load_t
            peak_mb = gpu_memory_mb()
            session.run(clips[0][1], language="id", timestamps="none")
            for clip_id, audio in clips:
                start = time.perf_counter()
                result = session.run(audio, language="id", timestamps="none")
                latencies.append((time.perf_counter() - start) * 1000)
                texts[clip_id] = result.text.strip()
                used = gpu_memory_mb()
                if used is not None:
                    peak_mb = max(peak_mb or used, used)
        backend = model.backend
    ref_words = sum(len(norm(REF[k])) for k in REF)
    errors = sum(wer(REF[k], texts[k]) * len(norm(REF[k])) for k in REF)
    model_mb = os.path.getsize(path) / 1e6
    delta = None if baseline_mb is None or peak_mb is None else max(0, peak_mb - baseline_mb)
    print(
        f"{label:10} {backend:7} file={model_mb:.0f}MB load={load_s:.2f}s "
        f"median={statistics.median(latencies):.0f}ms "
        f"WER={errors / ref_words * 100:.1f}% "
        f"VRAM_delta~={delta if delta is not None else 'n/a'}MB",
        flush=True,
    )
    for clip_id in REF:
        print(f"  {clip_id}: {texts[clip_id]} (WER {wer(REF[clip_id], texts[clip_id])*100:.1f}%)", flush=True)


if __name__ == "__main__":
    clips = load_clips()
    wanted = sys.argv[1:]
    print(f"clips={len(clips)}  backend=Vulkan  language=id", flush=True)
    for name, path in MODELS:
        if wanted and name not in wanted:
            continue
        benchmark(name, path, clips)
