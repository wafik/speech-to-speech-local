"""Pull a handful of short FLEURS Indonesian validation clips for the STT benchmark."""
import json, os, urllib.request

OUT = os.path.join(os.path.dirname(__file__), "audio")
os.makedirs(OUT, exist_ok=True)


def get(url):
    return urllib.request.urlopen(url, timeout=90).read()


rows = json.loads(get(
    "https://datasets-server.huggingface.co/rows"
    "?dataset=google%2Ffleurs&config=id_id&split=validation&offset=0&length=60"
))["rows"]

# Short sentences only: a real voice turn is 2-4 s, not a 12 s monologue.
cands = sorted(
    ((len(r["row"]["transcription"]), r) for r in rows),
    key=lambda t: t[0],
)[:8]

for _, r in cands:
    row = r["row"]
    dst = os.path.join(OUT, f"{row['id']}.wav")
    with open(dst, "wb") as f:
        f.write(get(row["audio"][0]["src"]))
print(json.dumps([{"id": r["row"]["id"], "text": r["row"]["transcription"]} for _, r in cands],
                 ensure_ascii=False, indent=1))
