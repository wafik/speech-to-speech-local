# speech-to-speech (RTX 5060 Ti 16GB) — thin wrapper over run.ps1
# Recipes run in Git Bash, which ships with the Claude/git install here.
PS := powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1
VENV := speech-to-speech/.venv/Scripts/python.exe
HF := speech-to-speech/.venv/Scripts/hf.exe

# Full-SGLang stack runs inside WSL (Ubuntu-2). scripts/wsl.sh lives on the
# Windows side; WSL_RUN copies it to ext4 and runs it there (executing straight
# off 9p is slow and mangles permissions). MSYS_NO_PATHCONV keeps Git Bash from
# rewriting the /root paths into C:/Program Files/...
WSL_DISTRO := Ubuntu-2
# This checkout is on E:. Override if you move it: make wsl WSL_WIN_PATH=/mnt/d/x
WSL_WIN_PATH ?= /mnt/e/test/spech-to-spech
WSL_RUN = MSYS_NO_PATHCONV=1 wsl.exe -d $(WSL_DISTRO) bash -c \
	"cp $(WSL_WIN_PATH)/scripts/wsl.sh /root/wsl.sh && bash /root/wsl.sh $(1)"

.DEFAULT_GOAL := run
.PHONY: run english 4b 9b serve multi indo indo-serve voice router setup setup-q4 setup-gguf models stop help \
        stt hf wsl wsl-setup wsl-status wsl-stop router router-low tunnel

run:            ## Start a voice session (Qwen3-ASR Q5 GGUF + gemma4:e4b + Supertonic)
	$(PS)

english:        ## English only: Parakeet + Kokoro (lower latency)
	$(PS) -English

4b:             ## qwen3.5:4b instead of gemma4:e4b
	$(PS) -Model qwen3.5:4b

9b:             ## qwen3.5:9b (+2.5GB VRAM, marginal gain)
	$(PS) -Model qwen3.5:9b

indo:           ## Same as `run`, but pins the TTS language to Indonesian
	$(PS) -Lang id

voice:          ## Pick a Supertonic voice: make voice VOICE=F3 (M1-M5 male, F1-F5 female)
	$(PS) -Voice $(or $(VOICE),F1)

stt:            ## GGUF STT quant: make stt STT_QUANT=Q8_0|Q6_K (default Q5_K_M)
	$(PS) -SttQuant $(or $(STT_QUANT),Q5_K_M)

hf:             ## HF bf16 Qwen3-ASR instead of the GGUF (1.57 GB, ~2x VRAM)
	$(PS) -Stt qwen3-asr

# Alternative LLM: 9router on :20128 instead of a local Ollama model. Same
# pipeline, same STT/TTS - only the text stage changes. run.ps1 finds the
# router's key itself (9R_KEY, then the Claude Code config); leave 9R_KEY empty
# unless you want a different one.
9R_BASE ?= http://127.0.0.1:20128/v1
9R_MODEL ?= 9r/routers9/nemotron-3.5-lightning
9R_KEY ?=
router:         ## Realtime server, LLM on 9router (9R_MODEL=..., optional 9R_KEY=...)
	$(PS) -Mode serve -Model "$(9R_MODEL)" -ApiBase "$(9R_BASE)" -EnableLlmProxy $(if $(9R_KEY),-ApiKey "$(9R_KEY)")

# Low-VRAM laptop preset: 2 GB GPU, CPU-only TTS. Turns are ~15 s end to end
# here because Supertonic runs on the same CPU the STT already saturates; the
# low-latency flags stop the LLM writing three sentences before TTS starts.
# No -EnableLlmProxy: this is the shape to run behind a tunnel.
router-low:     ## `router` + low-latency tuning, text proxy off (tunnel-safe)
	$(PS) -Mode serve -Model "$(9R_MODEL)" -ApiBase "$(9R_BASE)" -LowLatency $(if $(9R_KEY),-ApiKey "$(9R_KEY)")

# Expose the realtime server on voice.<domain> through the voice-speech tunnel.
CF_TUNNEL ?= voice-speech
tunnel:         ## Run the voice-speech tunnel (needs `make router-low` in another shell)
	cloudflared tunnel run --url http://localhost:8765 $(CF_TUNNEL)

indo-serve:     ## Indonesian realtime server for other agents
	$(PS) -Lang id -Mode serve

serve:          ## Realtime server for other agents (ws://127.0.0.1:8765/v1/realtime)
	$(PS) -Mode serve

multi:          ## Same server, 3 concurrent sessions (PIPELINES=n to change)
	$(PS) -Mode serve -Pipelines $(or $(PIPELINES),3)

setup:          ## One-time: clone upstream, build venv, CUDA torch, extras, LLM, STT
	@test -d speech-to-speech || git clone https://github.com/huggingface/speech-to-speech
	uv sync --python 3.11 --project speech-to-speech
	uv pip install --python $(VENV) "torch==2.11.0" "torchaudio==2.11.0" \
		--index-url https://download.pytorch.org/whl/cu128 \
		--reinstall-package torch --reinstall-package torchaudio
	# kokoro+spacy pinned: kokoro drags in spacy 4.0.0.dev3, which breaks numpy
	uv pip install --python $(VENV) "kokoro>=0.9.2" "spacy>=3.8.4,<4" "supertonic>=1.3.1"
	ollama pull gemma4:e4b
	$(MAKE) setup-q4
	$(MAKE) setup-gguf

setup-q4:       ## Fetch the q4 Supertonic graphs (default TTS, ~1.8x faster)
	$(VENV) bench/fetch_tts_variant.py nadare/supertonic-3-onnx-q4 onnx models/supertonic-3-q4

# STT default: Qwen3-ASR-0.6B Q5_K_M via transcribe.cpp (Vulkan here).
# Q5_K_M matches bf16 WER on the Indonesian clips at 645 MB instead of 1.57 GB.
# Pick another level with STT_QUANT=Q6_K|Q8_0; Q4_K_M measurably regressed (4.8%).
STT_QUANT ?= Q5_K_M
GGUF_DIR := models/qwen-asr-gguf
GGUF_REPO := handy-computer/Qwen3-ASR-0.6B-gguf

setup-gguf:     ## Fetch the Qwen3-ASR GGUF + transcribe.cpp runtime (STT default)
	uv pip install --python $(VENV) "transcribe-cpp==0.4.1"
	@test -f $(GGUF_DIR)/Qwen3-ASR-0.6B-$(STT_QUANT).gguf || \
		$(HF) download $(GGUF_REPO) "Qwen3-ASR-0.6B-$(STT_QUANT).gguf" --local-dir $(GGUF_DIR)
	@test -f $(GGUF_DIR)/Qwen3-ASR-0.6B-$(STT_QUANT).gguf && \
		echo "STT ready: $(GGUF_DIR)/Qwen3-ASR-0.6B-$(STT_QUANT).gguf"

models:         ## Show installed Ollama models, VRAM, and audio devices
	ollama ls
	nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
	$(VENV) -c "import sounddevice as sd; print(sd.query_devices())"

stop:           ## Kill a running session (the STT sidecar exits with it)
	taskkill //F //IM speech-to-speech.exe

# --- Full SGLang stack, all inside WSL -------------------------------------
# Copies scripts/wsl.sh to /root and runs it there: one file, no drift between
# the repo copy and the one WSL executes.
wsl:            ## All-SGLang stack in WSL (ASR+LLM on GPU, TTS on CPU) + browser UI
	$(call WSL_RUN,serve)

wsl-setup:      ## One-time WSL setup: venvs, models, UI page
	$(call WSL_RUN,setup)

wsl-status:     ## What is listening in WSL, plus VRAM
	$(call WSL_RUN,status)

wsl-stop:       ## Stop every WSL process from `make wsl`
	$(call WSL_RUN,stop)

help:
	@echo "make             run (Qwen3-ASR Q5 GGUF + gemma4:e4b + Supertonic, auto language)"
	@echo "make english     English only: Parakeet + Kokoro (lower latency)"
	@echo "make 4b          run with qwen3.5:4b"
	@echo "make 9b          run with qwen3.5:9b"
	@echo "make indo        pin TTS to Indonesian"
	@echo "make voice       Supertonic voice: VOICE=F3 (M1-M5 male, F1-F5 female)"
	@echo "make stt         STT quant: STT_QUANT=Q8_0|Q6_K (default Q5_K_M)"
	@echo "make hf          HF bf16 Qwen3-ASR instead of the GGUF"
	@echo "make router      9router LLM instead of Ollama (9R_MODEL, 9R_KEY)"
	@echo "make router-low  same, low-latency preset for small GPUs / CPU TTS"
	@echo "make tunnel      run the cloudflared tunnel (CF_TUNNEL=<name>, default voice-speech)"
	@echo "make setup-gguf  fetch the Qwen3-ASR GGUF + transcribe.cpp (STT)"
	@echo "make setup-q4    fetch the q4 Supertonic graphs if missing"
	@echo "make indo-serve  Indonesian realtime server"
	@echo "make setup       install everything from scratch"
	@echo "make models      list models / VRAM / audio devices"
	@echo "make stop        kill running session"
	@echo "make wsl         full-SGLang stack in WSL + browser UI on :7860"
	@echo "make wsl-setup   one-time WSL setup (venvs, models)"
	@echo "make wsl-status  what is up in WSL"
	@echo "make wsl-stop    stop the WSL stack"
