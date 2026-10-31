# speech-to-speech-local

Local, offline voice chat on one consumer GPU (tested on an RTX 5060 Ti 16 GB).
A thin launcher around [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) —
the pipeline itself is upstream, unmodified. This repo adds the setup, the model
choices that actually fit in 16 GB, and the benchmarks behind those choices.

## Stack

```
mic -> Silero VAD -> Qwen3-ASR -> gemma4:e4b (Ollama) -> Supertonic -> speaker
```

There is also a [full-SGLang variant](#full-sglang-all-inside-wsl) that runs the
same pipeline entirely inside WSL, for comparison.

| Stage | Default | Why |
|---|---|---|
| VAD | Silero v5 | upstream default, 16 kHz |
| STT | `Qwen/Qwen3-ASR-0.6B-hf` | 30 languages incl. Indonesian + Malay, auto language ID, handles Indonesian/English code-switching |
| LLM | `gemma4:e4b` @ Ollama | ~0.31 s/turn warm with reasoning off |
| TTS | Supertonic | 32 languages, 66M params, ONNX on CPU (no VRAM) |
| English-only | Parakeet TDT + Kokoro | faster per turn, but 25 European languages / 8 voices |

Indonesian needs the Qwen3-ASR + Supertonic pair. Parakeet TDT v3 covers 25
European languages and has no Indonesian; Kokoro has no Indonesian voice either.
Swapping only one stage produces the right words in the wrong phonemes.

## Requirements

- Windows + PowerShell 5.1 (the launcher targets it; ASCII-only, no BOM)
- NVIDIA GPU, CUDA 12.8 wheels (adjust `setup` for another CUDA version)
- [uv](https://docs.astral.sh/uv/) >= 0.13.4 — `uv self update`
- [Ollama](https://ollama.com/) on `127.0.0.1:11434`
- git, make, and a C++ build toolchain for `ctranslate2` if you use faster-whisper

## Setup

```bash
git clone https://github.com/wafik/speech-to-speech-local
cd speech-to-speech-local
make setup
```

`setup` clones upstream into `speech-to-speech/`, builds the venv, installs the
CUDA torch wheels, the TTS extras, and pulls the LLM.

Model checkpoints download on first run. To keep them off the system drive:

```powershell
setx HF_HOME D:\hf_cache      # Hugging Face models
setx OLLAMA_MODELS D:\ollama  # Ollama blobs
```

## Run

```bash
make              # Qwen3-ASR + gemma4:e4b + Supertonic, language auto-detected
make english      # Parakeet + Kokoro, lower latency, English only
make indo         # same as make, pins the TTS voice to Indonesian
make 4b           # qwen3.5:4b instead of gemma4:e4b
make 9b           # qwen3.5:9b
make serve        # headless realtime server, 3 concurrent sessions
make stop         # kill a running session
```

`run.ps1` also takes `-Online` to allow Hugging Face network calls; the default
is fully offline once the checkpoints are cached.

## Full SGLang (all inside WSL)

The Ollama stack above is the reference. This second stack replaces every GPU
stage with SGLang and runs entirely in WSL — same upstream pipeline, different
serving layer, so you can compare them side by side.

```
mic -> Silero VAD -> SGLang :8000  Qwen3-ASR-0.6B
                 -> SGLang :30000 gemma-4-E4B int4 (W4A16 QAT)
                 -> Supertonic (CPU, 66M) -> speaker
```

| Service | Port | Notes |
|---|---|---|
| Qwen3-ASR | 8000 | `/v1/audio/transcriptions`, chunked streaming adapter |
| gemma-4-E4B | 30000 | 4-bit QAT, `--reasoning_effort none` |
| pipeline | 8765 | `ws://127.0.0.1:8765/v1/realtime`, 3 slots |
| browser UI | 7860 | click, allow the mic, talk |

```bash
make wsl-setup   # once: two venvs, ~30 GB of weights, UI page
make wsl         # start everything, prints the UI URL
make wsl-status  # ports + VRAM + loaded model ids
make wsl-stop    # stop all four
```

Then open <http://localhost:7860/>.

**Two venvs are mandatory.** `speech-to-speech` pins `openai==2.28.0`; SGLang
0.5.22 pins `openai==2.6.1`. They cannot share an environment.

**VRAM, 16 GB card:**

| | Weights | KV | Total |
|---|---|---|---|
| Qwen3-ASR 0.6B bf16 | 1.57 GB | 0.54 GB | ~2.1 GB |
| gemma-4-E4B W4A16 | 12.87 GB | 0.32 GB | ~13.2 GB |
| Supertonic | — | — | CPU |

That leaves ~1.5 GB. `gemma-4-E4B` at bf16 is **16 GB** and does not fit
alongside ASR — use the QAT checkpoint. `--mem-fraction-static` must be 0.90
for the LLM and 0.15 for ASR; lower values leave no room for the KV pool.

Three things SGLang gets wrong out of the box, all handled in `scripts/wsl.sh`:

1. **`Qwen3-ASR-0.6B-hf` is not loadable.** Its `config.json` is flat
   (`audio_config`/`text_config` at top level) but SGLang's `Qwen3ASRConfig`
   reads `thinker_config`. It silently falls back to a default `Qwen3Config`
   (4096 hidden, 32 layers), tries to allocate ~14.8 GB, and dies on the KV
   cache. Use `Qwen/Qwen3-ASR-0.6B`, not the `-hf` mirror.
2. **The 4-bit gemma checkpoint crashes on load.** Its audio tower is fully
   listed in the checkpoint's quant `ignore` list, but `gemma4_mm.py` still
   passes `quant_config` down, so `ClippableGLUParallelLinear` builds a
   wrapper with no `.weight`. The script patches that call site to
   `quant_config=None` (idempotent, one file).
3. **Start the servers serially.** Both load into the same 16 GB; racing them
   OOMs ASR.

Measured on the same Indonesian clip (4.8 s in):

| | Ollama stack | Full SGLang |
|---|---|---|
| STT | 136 ms | 170 ms |
| LLM | — | 420 to 630 ms |
| Reply audio | — | 8 s |

Both pipelines are the same code, so the only real difference is the serving
layer. The Ollama path stays the default because it needs no WSL, no second
venv, and no source patch.

## OpenAI-compatible API

`make serve` exposes the same protocol another agent can talk to:

| Endpoint | Use |
|---|---|
| `ws://127.0.0.1:8765/v1/realtime` | voice (OpenAI Realtime over WebSocket) |
| `POST /v1/realtime/calls` | voice (WebRTC) |
| `POST /v1/chat/completions` | text, borrows the same LLM (`--enable_llm_proxy`) |
| `GET /v1/pool` | slot status |

There is **no** `/v1/audio/speech` or `/v1/audio/transcriptions` — audio only
flows through the realtime session.

`client_demo.py` is a working reference for both. Two things bite people:

1. **`reasoning_effort: "none"` is mandatory.** The proxy forwards it verbatim.
   Without it `gemma4:e4b` spends 2.1 s thinking per turn and `qwen3.5:4b` 14 s.
2. **Declare 16 kHz output, or declare nothing.** The pipeline resamples Kokoro
   24 kHz down to its native 16 kHz. Asking for 24 kHz makes the server upsample
   back — a 24k to 16k to 24k round trip that discards the 8 to 12 kHz band and
   costs 1.5x bandwidth for worse audio. Playing 16 kHz data as if it were 24 kHz
   makes speech 1.5x fast and high-pitched.

```python
# text agent
llm = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="unused")
llm.chat.completions.create(
    model="ignored",                       # server forces --model_name
    messages=[{"role": "user", "content": "hi"}],
    extra_body={"reasoning_effort": "none"},
)
```

## Benchmarks

Measured on this machine, 8 FLEURS Indonesian validation clips.

### STT (WER after normalisation, latency per turn)

| | Qwen3-ASR-0.6B | faster-whisper medium-id |
|---|---|---|
| Corpus WER | 2.4 % | 2.4 % |
| 6.5 to 10.9 s clip | 723 ms | **309 ms** |
| 2.5 s turn | 136 ms | **129 ms** |
| Peak VRAM | 1.67 GB | ~0.8 GB (int8) |
| Code-switching | **yes** | no, forces Indonesian |

The WER tie is 8 clips with identical errors — not a real ranking. The latency
split is real: Whisper is constant-time, Qwen3-ASR's decoder grows with audio
length. Use faster-whisper if turns routinely run past 5 s.

```bash
python bench/fetch_clips.py                          # needs the pipeline venv
python bench/bench_stt.py                            # qwen|fw|both
python bench/wer.py
python bench/e2e_id.py bench/audio_short.wav         # end-to-end against `make serve`
```

### LLM (identical prompt, via `/v1/chat/completions`)

| Model | `reasoning_effort` | Latency | Tokens |
|---|---|---|---|
| gemma4:e4b | `none` | **0.31 s** | 37 |
| qwen3.5:4b | `none` | 0.59 s | 49 |
| gemma4:e4b | unset | 2.11 s | 318 |
| qwen3.5:4b | unset | 14.28 s | 1096 |

### TTS

Supertonic runs on CPU via ONNX: RTF 0.18 to 0.20, so about 820 ms for 4 s of
audio. The 0.005 RTF on its model card is an RTX 4090, not what you get here.

### End to end

```
in   : "Halo, bisa bantu saya setup Docker container untuk aplikasi ini?"
STT  : "halo bisa bantu saya setup docker container untuk aplikasi ini nanti deploynya ke production ya"
BOT  : "Tentu, saya bisa bantu Anda mengatur kontainer Docker tersebut."
TTFA : 2.1 s warm   (STT 0.14 s + LLM 0.31 s + TTS first chunk)
```

## Caveats

- **Session slots recover slowly.** A closed session holds its slot about 1.4 s.
  Reconnecting inside that window returns `1008 All session slots are in use`.
  `serve` defaults to 3 slots to absorb it.
- **The LLM proxy has no auth and no rate limit.** Bind it to localhost or put a
  gateway in front. Do not expose it to a network you do not control.
- **`gemma4:e4b` is 7.5B parameters** (Q4_K_M), despite the name — "E4B" means
  *effective* 4B.
- **VRAM:** LLM ~3.1 GB + STT 1.7 GB + VAD/TTS overhead fits in 16 GB with room
  for 3 pipeline slots. Each extra slot loads its own STT and TTS.

## License

The wrapper and benchmarks here are MIT. Upstream `speech-to-speech` is MIT;
individual model checkpoints carry their own licences — check them before
commercial use (`facebook/mms-tts-ind`, for instance, is CC-BY-NC 4.0).
