#!/usr/bin/env bash
# Full-SGLang stack inside WSL (Ubuntu-2).
#
#   mic -> VAD -> SGLang :8000  Qwen3-ASR-0.6B
#              -> SGLang :30000 gemma-4-E4B int4 (W4A16 QAT)
#              -> Supertonic CPU 66M -> speaker
#
# Called from the Makefile (`make wsl`), or directly via WSL. Everything runs on
# ext4 under /root so it never crawls over the 9p /mnt/c mount.
set -euo pipefail

VENV_SGL="${VENV_SGL:-/root/venv_netra}"   # sglang 0.5.22, torch 2.14
VENV_S2S="${VENV_S2S:-/root/s2s/.venv}"    # speech-to-speech, openai 2.28
HF="${HF_HOME:-/root/.cache/huggingface}"

ASR_REPO="$HF/hub/models--Qwen--Qwen3-ASR-0.6B"
LLM_REPO="$HF/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct"
ASR_PORT=8000
LLM_PORT=30000
S2S_PORT=8765
UI_PORT=7860

snap() { ls -d "$1"/snapshots/*/ 2>/dev/null | head -1; }

wait_http() {  # wait_http <port> <label>
    for _ in $(seq 1 150); do
        if curl -sf -o /dev/null "http://127.0.0.1:$1/v1/models" 2>/dev/null; then return 0; fi
        sleep 2
    done
    echo "  $2 did not come up on :$1 - see /root/logs/$2.log" >&2
    return 1
}

# The 4-bit gemma checkpoint leaves the audio tower unquantized, but SGLang
# still hands it a quant_config and builds wrappers with no .weight. Audio is
# unused in this pipeline, so pass None there.
patch_gemma4() {
    local f="$VENV_SGL/lib/python3.12/site-packages/sglang/srt/models/gemma4_mm.py"
    [ -f "$f" ] || return 0
    grep -q "audio tower is fully in the checkpoint" "$f" && return 0
    python3 - "$f" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
for old, new in (
    ("""self.audio_tower = Gemma4AudioEncoder(
                    config=config.audio_config,
                    quant_config=quant_config,""",
     """self.audio_tower = Gemma4AudioEncoder(
                    config=config.audio_config,
                    # audio tower is fully in the checkpoint's quant ignore
                    # list; quant_config builds wrappers without .weight.
                    quant_config=None,"""),
    ("""self.embed_audio = Gemma4MultimodalEmbedder(
                    config=config.audio_config,
                    config.text_config,
                    quant_config=quant_config,""",
     """self.embed_audio = Gemma4MultimodalEmbedder(
                    config=config.audio_config,
                    config.text_config,
                    quant_config=None,"""),
):
    assert s.count(old) == 1, f"patch target not found in {p}"
    s = s.replace(old, new)
open(p, "w").write(s)
PY
    echo "  patched gemma4_mm.py (audio tower unquantized)"
}

start() {  # start <name> <logfile> <cmd...>
    local name=$1 log=$2; shift 2
    if ss -ltn 2>/dev/null | grep -q ":$name\b"; then
        echo "  :$name already listening"; return 0
    fi
    setsid nohup "$@" > "$log" 2>&1 < /dev/null &
    disown
}

cmd_serve() {
    mkdir -p /root/logs
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$HF"

    local asr llm
    asr=$(snap "$ASR_REPO"); llm=$(snap "$LLM_REPO")
    [ -n "$asr" ] || { echo "missing $ASR_REPO - run 'make wsl-setup'" >&2; exit 1; }
    [ -n "$llm" ] || { echo "missing $LLM_REPO - run 'make wsl-setup'" >&2; exit 1; }

    echo "ASR  :8000  Qwen3-ASR-0.6B"
    start $ASR_PORT /root/logs/asr.log "$VENV_SGL/bin/python" -u -m sglang.launch_server \
        --model-path "$asr" --host 0.0.0.0 --port $ASR_PORT \
        --mem-fraction-static 0.15 --attention-backend triton \
        --disable-custom-all-reduce --served-model-name qwen3-asr

    # Serialize: both load weights into the same 16 GB, and racing them OOMs
    # the smaller one. ASR takes ~30 s to reach ready.
    wait_http $ASR_PORT asr

    echo "LLM  :$LLM_PORT gemma-4-E4B int4"
    patch_gemma4
    start $LLM_PORT /root/logs/llm.log "$VENV_SGL/bin/python" -u -m sglang.launch_server \
        --model-path "$llm" --host 0.0.0.0 --port $LLM_PORT \
        --mem-fraction-static 0.90 --attention-backend triton \
        --disable-custom-all-reduce --context-length 8192 \
        --served-model-name gemma-4-e4b

    wait_http $LLM_PORT llm

    echo "PIPE :$S2S_PORT speech-to-speech (3 pipelines)"
    cd /root/s2s
    start $S2S_PORT /root/logs/pipeline.log "$VENV_S2S/bin/speech-to-speech" serve \
        --llm_backend chat-completions --model_name gemma-4-e4b \
        --responses_api_base_url "http://127.0.0.1:$LLM_PORT/v1" \
        --responses_api_api_key none --responses_api_reasoning_effort none \
        --stt openai --openai_stt_base_url "http://127.0.0.1:$ASR_PORT/v1" \
        --openai_stt_model qwen3-asr \
        --tts supertonic --enable_llm_proxy --num_pipelines 3

    echo "UI   :$UI_PORT http://localhost:$UI_PORT/"
    start $UI_PORT /root/logs/ui.log "$VENV_S2S/bin/python" -m http.server $UI_PORT \
        --bind 0.0.0.0 --directory /root/s2s/ui

    for _ in $(seq 1 60); do
        ss -ltn | grep -q ":$S2S_PORT\b" && break
        sleep 2
    done

    echo
    echo "Speak at http://localhost:$UI_PORT/  (mic permission required)"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
}

cmd_stop() {
    local pats=("sglang.launch_server" "speech-to-speech serve" "http.server $UI_PORT")
    for p in "${pats[@]}"; do pkill -f "$p" 2>/dev/null || true; done
    # sglang traps SIGTERM and can take a while to unwind its schedulers
    for _ in $(seq 1 10); do
        pgrep -f "sglang.launch_server|speech-to-speech serve" > /dev/null || break
        sleep 1
    done
    for p in "${pats[@]}"; do pkill -9 -f "$p" 2>/dev/null || true; done
    sleep 1
    echo "stopped"
}

cmd_status() {
    for p in $ASR_PORT $LLM_PORT $S2S_PORT $UI_PORT; do
        if ss -ltn 2>/dev/null | grep -q ":$p\b"; then echo "  :$p listening"; else echo "  :$p down"; fi
    done
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    echo
    curl -s "http://127.0.0.1:$ASR_PORT/v1/models" | head -c 200; echo
    curl -s "http://127.0.0.1:$LLM_PORT/v1/models" | head -c 200; echo
}

# One-time: venvs, models, npm-free UI. Idempotent.
cmd_setup() {
    command -v nvidia-smi > /dev/null || { echo "no CUDA in WSL - run 'nvidia-smi' first" >&2; exit 1; }

    echo "== sglang venv =="
    [ -d "$VENV_SGL" ] || python3 -m venv "$VENV_SGL"
    "$VENV_SGL/bin/pip" install -q --upgrade pip
    "$VENV_SGL/bin/pip" install -q "sglang[all]==0.5.22"

    echo "== speech-to-speech venv =="
    [ -d /root/s2s ] || git clone https://github.com/huggingface/speech-to-speech /root/s2s
    [ -d "$VENV_S2S" ] || python3 -m venv "$VENV_S2S"
    "$VENV_S2S/bin/pip" install -q --upgrade pip
    "$VENV_S2S/bin/pip" install -q -e /root/s2s "supertonic>=1.3.1"

    echo "== models (ext4, ~30 GB) =="
    "$VENV_SGL/bin/hf" download Qwen/Qwen3-ASR-0.6B
    "$VENV_SGL/bin/hf" download google/gemma-4-E4B-it-qat-w4a16-ct

    echo "== ui =="
    mkdir -p /root/s2s/ui
    # This script is copied to /root, so it cannot locate the repo itself.
    local ui="${S2S_REPO:-/mnt/e/test/spech-to-spech}/ui/index.html"
    cp "$ui" /root/s2s/ui/index.html

    patch_gemma4
    echo "setup done"
}

case "${1:-serve}" in
    serve)  cmd_serve ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    setup)  cmd_setup ;;
    *)      echo "usage: $0 {serve|stop|status|setup}" >&2; exit 2 ;;
esac
