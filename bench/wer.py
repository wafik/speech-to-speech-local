"""WER + latency comparison from bench_result.json against the FLEURS references."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

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


def norm(s):
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^\w\s]|_", " ", s)
    return s.split()


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


if __name__ == "__main__":
    data = json.load(open(os.path.join(HERE, "bench_result.json"), encoding="utf-8"))
    print(f"{'clip':>6} | {'qwen3 0.6B':>10} | {'fw medium-id':>12}")
    totals = {"qwen": [0, 0], "fw": [0, 0]}
    for cid, ref in REF.items():
        row = []
        for key in ("qwen", "fw"):
            hyp = data.get(key, {}).get("texts", {}).get(cid, "")
            e = wer(ref, hyp)
            totals[key][0] += e * len(norm(ref))
            totals[key][1] += len(norm(ref))
            row.append(f"{e*100:9.1f}%")
        print(f"{cid:>6} | {row[0]:>10} | {row[1]:>12}")
    for key, label in (("qwen", "qwen3-asr-0.6B"), ("fw", "faster-whisper medium-id")):
        s = data[key]
        print(f"\n{label}: corpus WER {totals[key][0]/totals[key][1]*100:.1f}% | "
              f"median {s['median_ms']} ms | peak VRAM {s['vram_gb']:.2f} GB | load {s['load_s']}s")
