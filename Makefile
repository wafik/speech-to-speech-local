# speech-to-speech (RTX 5060 Ti 16GB) — thin wrapper over run.ps1
# Recipes run in Git Bash, which ships with the Claude/git install here.
PS := powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1
VENV := speech-to-speech/.venv/Scripts/python.exe

.DEFAULT_GOAL := run
.PHONY: run english 4b 9b serve multi indo indo-serve setup models stop help

run:            ## Start a voice session (Qwen3-ASR + gemma4:e4b + Supertonic)
	$(PS)

english:        ## English only: Parakeet + Kokoro (lower latency)
	$(PS) -English

4b:             ## qwen3.5:4b instead of gemma4:e4b
	$(PS) -Model qwen3.5:4b

9b:             ## qwen3.5:9b (+2.5GB VRAM, marginal gain)
	$(PS) -Model qwen3.5:9b

indo:           ## Same as `run`, but pins the TTS language to Indonesian
	$(PS) -Lang id

indo-serve:     ## Indonesian realtime server for other agents
	$(PS) -Lang id -Mode serve

serve:          ## Realtime server for other agents (ws://127.0.0.1:8765/v1/realtime)
	$(PS) -Mode serve

multi:          ## Same server, 3 concurrent sessions (PIPELINES=n to change)
	$(PS) -Mode serve -Pipelines $(or $(PIPELINES),3)

setup:          ## One-time: clone upstream, build venv, CUDA torch, extras, LLM
	@test -d speech-to-speech || git clone https://github.com/huggingface/speech-to-speech
	uv sync --python 3.11 --project speech-to-speech
	uv pip install --python $(VENV) "torch==2.11.0" "torchaudio==2.11.0" \
		--index-url https://download.pytorch.org/whl/cu128 \
		--reinstall-package torch --reinstall-package torchaudio
	# kokoro+spacy pinned: kokoro drags in spacy 4.0.0.dev3, which breaks numpy
	uv pip install --python $(VENV) "kokoro>=0.9.2" "spacy>=3.8.4,<4" "supertonic>=1.3.1"
	ollama pull gemma4:e4b

models:         ## Show installed Ollama models, VRAM, and audio devices
	ollama ls
	nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
	$(VENV) -c "import sounddevice as sd; print(sd.query_devices())"

stop:           ## Kill a running session
	taskkill //F //IM speech-to-speech.exe

help:
	@echo "make             run (Qwen3-ASR + gemma4:e4b + Supertonic, auto language)"
	@echo "make english     English only: Parakeet + Kokoro (lower latency)"
	@echo "make 4b          run with qwen3.5:4b"
	@echo "make 9b          run with qwen3.5:9b"
	@echo "make indo        pin TTS to Indonesian"
	@echo "make indo-serve  Indonesian realtime server"
	@echo "make setup       install everything from scratch"
	@echo "make models      list models / VRAM / audio devices"
	@echo "make stop        kill running session"
