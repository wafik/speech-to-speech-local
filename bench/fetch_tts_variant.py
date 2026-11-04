"""Download one Supertonic-3 ONNX variant into a directory.

    python bench/fetch_tts_variant.py <hf-repo> <subdir-or-.> <out-dir>

Voice styles are copied from the installed full-precision cache - they are just
speaker embeddings and do not change with graph precision.
"""
import os
import shutil
import sys

from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
REPO, SUB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = OUT if os.path.isabs(OUT) else os.path.join(os.path.dirname(HERE), OUT)

root = snapshot_download(repo_id=REPO)
src = os.path.join(root, SUB) if SUB not in (".", "") else root
print("from:", src)

dst = os.path.join(OUT, "onnx")
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if not f.endswith((".onnx", ".json")):
        continue
    # HF snapshots are symlinks into blobs/; realpath resolves them to the bytes.
    shutil.copyfile(os.path.realpath(os.path.join(src, f)), os.path.join(dst, f))
    print(f"  {f}  {os.path.getsize(os.path.join(dst, f))/1e6:.1f}MB")

# Voice styles: reuse whatever the supertonic package already cached.
styles_src = os.path.join(os.path.expanduser("~"), ".cache", "supertonic3", "voice_styles")
if os.path.isdir(styles_src):
    styles_dst = os.path.join(OUT, "voice_styles")
    os.makedirs(styles_dst, exist_ok=True)
    for f in os.listdir(styles_src):
        shutil.copyfile(os.path.join(styles_src, f), os.path.join(styles_dst, f))
    print(f"  voice_styles: {len(os.listdir(styles_dst))}")
else:
    print("  WARNING: no voice styles at", styles_src)

print("ready:", OUT)
