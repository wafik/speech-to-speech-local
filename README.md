# speech-to-speech-local

Local, offline voice chat on one consumer GPU (tested on an RTX 5060 Ti 16 GB).
A thin launcher around [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) —
the pipeline itself is upstream, unmodified. This repo adds the setup, the model
choices that actually fit in 16 GB, and the benchmarks behind those choices.

## Stack

```
mic -> Silero VAD -> Qwen3-ASR Q5_K_M (GGUF) -> gemma4:e4b (Ollama) -> Supertonic -> speaker
```

There is also a [full-SGLang variant](#full-sglang-all-inside-wsl) that runs the
same pipeline entirely inside WSL, for comparison.

| Stage | Default | Why |
|---|---|---|
| VAD | Silero v5 | upstream default, 16 kHz |
| STT | Qwen3-ASR 0.6B **Q5_K_M GGUF** | 645 MB on disk, Vulkan via transcribe.cpp, 30 languages incl. Indonesian + Malay, auto language ID, code-switching. Same WER as bf16 on the test clips |
| LLM | `gemma4:e4b` @ Ollama | ~0.31 s/turn warm with reasoning off |
| TTS | Supertonic q4 | 31 languages, ONNX on CPU (no VRAM), 127 MB |
| Voice | `F1` | 5 female (F1-F5) and 5 male (M1-M5) styles |
| English-only | Parakeet TDT + Kokoro | faster per turn, but 25 European languages / 8 voices |

Indonesian needs the Qwen3-ASR + Supertonic pair. Parakeet TDT v3 covers 25
European languages and has no Indonesian; Kokoro has no Indonesian voice either.
Swapping only one stage produces the right words in the wrong phonemes.

## Requirements

- Windows + PowerShell 5.1 (the launcher targets it; ASCII-only, no BOM)
- NVIDIA GPU, CUDA 12.8 wheels (adjust `setup` for another CUDA version)
- Vulkan driver for the GGUF STT (ships with current NVIDIA drivers; `vulkaninfo` lists the GPU)
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
CUDA torch wheels, the TTS extras, the STT runtime + weights, and pulls the LLM.

Model checkpoints download on first run. To keep them off the system drive:

```powershell
setx HF_HOME E:\models\hf     # Hugging Face models
setx OLLAMA_MODELS D:\ollama  # Ollama blobs
```

`run.ps1` runs offline by default, so `HF_HOME` must already hold every
checkpoint a stage needs. If a run dies with `OfflineModeIsEnabled`, either
pass `-Online` once or check the repo is in `<HF_HOME>\hub`.

## STT: Qwen3-ASR as a GGUF

The default speech recogniser is the **Q5_K_M** quantisation of
[`Qwen3-ASR-0.6B`](https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf)
run by [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
through its Vulkan backend. 645 MB on disk instead of 1.57 GB, and it measured
the same 2.4 % WER as the bf16 checkpoint on the Indonesian clips below.

```bash
make setup-gguf                      # runtime + Q5_K_M weights (one-time)
make setup-gguf STT_QUANT=Q8_0       # or Q6_K / Q8_0
```

What that target does, step by step:

```powershell
# 1. Python bindings. The wheel bundles the Vulkan/CPU native runtime,
#    so no separate transcribe-cli.exe is needed.
uv pip install --python speech-to-speech/.venv/Scripts/python.exe "transcribe-cpp==0.4.1"

# 2. Weights. They live in models/, not the HF cache.
speech-to-speech/.venv/Scripts/hf.exe download handy-computer/Qwen3-ASR-0.6B-gguf `
    "Qwen3-ASR-0.6B-Q5_K_M.gguf" --local-dir models/qwen-asr-gguf

# 3. Verify: 645,356,192 bytes, sha256 062a7bcb...16b54358
Get-FileHash models/qwen-asr-gguf/Qwen3-ASR-0.6B-Q5_K_M.gguf -Algorithm SHA256
```

Available files in that repo, and why Q5_K_M is the default:

| File | Size | sha256 (first 8) | Indonesian WER |
|---|---|---|---|
| `...-BF16.gguf` | 1.57 GB | `dadb8196` | 2.4 % |
| `...-Q8_0.gguf` | 850 MB | `f081b2d5` | 2.4 % |
| `...-Q6_K.gguf` | 690 MB | `3b051f10` | 2.4 % |
| **`...-Q5_K_M.gguf`** | **645 MB** | **`062a7bcb`** | **2.4 %** |
| `...-Q4_K_M.gguf` | 590 MB | `5b58f32a` | 4.8 % ← rejected |

Q4 is the only step that loses accuracy here; the published LibriSpeech numbers
move from 2.12 % to 2.26 %, but on these clips one more word goes wrong. Q5_K_M
is the last level that stays word-identical, so it wins on size-per-error.

### How it plugs into the pipeline

The upstream clone stays unmodified. `scripts/gguf_stt_server.py` is a small
OpenAI-compatible server (`POST /v1/audio/transcriptions`, `GET /health`), and
the pipeline's existing `--stt openai` backend talks to it:

```
speech-to-speech.exe --stt openai --openai_stt_base_url http://127.0.0.1:8011/v1
        |
        v
scripts/gguf_stt_server.py  (FastAPI + transcribe_cpp)  ->  Vulkan GPU
```

`run.ps1` starts that server on `127.0.0.1:8011` when it is not already
listening, waits for `/health`, and stops it when the pipeline exits. A server
you started yourself is left alone. One model instance serves all pipeline
slots, so `make multi` does not load three copies of the weights.

Run the server by hand when you want to poke at it:

```powershell
speech-to-speech\.venv\Scripts\python.exe -I scripts\gguf_stt_server.py `
    --model models\qwen-asr-gguf\Qwen3-ASR-0.6B-Q5_K_M.gguf --port 8011
```

```bash
curl -F file=@bench/audio/1515.wav -F model=qwen3-asr http://127.0.0.1:8011/v1/audio/transcriptions
# {"text":"Ini adalah CEP kelima Martelli dalam empat tahun.","language":"id-auto"}
```

`language` comes back as `<code>-auto` when the model detected it, matching the
upstream Qwen handler; send `-F language=id` to force one and get a bare `id`.

### Switching or falling back

```bash
make                                  # Q5_K_M GGUF (default)
make stt STT_QUANT=Q8_0               # other GGUF level; needs make setup-gguf STT_QUANT=... first
make hf                               # bf16 Qwen3-ASR through Transformers (1.57 GB, ~1.7 GB VRAM)
make english                          # Parakeet TDT + Kokoro, English only
```

`make hf` also returns to the HF cache check: the bf16 checkpoint has to be in
`HF_HOME`, and offline mode needs it cached first. The GGUF path never touches
the network at run time.

## Run

```bash
make              # Qwen3-ASR Q5 GGUF + gemma4:e4b + Supertonic, language auto-detected
make english      # Parakeet + Kokoro, lower latency, English only
make indo         # same as make, pins the TTS voice to Indonesian
make voice        # female voice by default; VOICE=F3 for another one
make stt          # STT_QUANT=Q8_0 or Q6_K instead of Q5_K_M
make hf           # bf16 Qwen3-ASR instead of the GGUF
make router       # 9router LLM instead of Ollama (see below)
make 4b           # qwen3.5:4b instead of gemma4:e4b
make 9b           # qwen3.5:9b
make serve        # headless realtime server, 3 concurrent sessions
make stop         # kill a running session
```

`run.ps1` also takes `-Online` to allow Hugging Face network calls; the default
is fully offline once the checkpoints are cached.

### Alternate LLM on 9router

`make router` points the text stage at [9router](https://127.0.0.1:20128/v1)
instead of Ollama. Same pipeline, same STT/TTS — only the LLM endpoint changes,
and the local Ollama server is not started.

```bash
make router                                    # nemotron-3.5-lightning on :20128
make router 9R_MODEL=9r/routers9/nemotron-3.5-super-vl
9R_KEY=sk-... make router                      # only to override the auto-found key
```

The key is resolved automatically: `9R_KEY`, then the token Claude Code already
uses for that endpoint in `~/.claude/settings.json`. Nothing is written anywhere.
9router must already be running (`npx 9router` or its background service).

Needs nothing but an OpenAI-compatible `/v1/chat/completions` that accepts
`reasoning_effort: "none"`. Anything else — vLLM, llama.cpp, LM Studio, a
hosted API — works the same way via `run.ps1 -ApiBase <url> -ApiKey <key>`.

#### Low-latency preset (small GPUs, CPU TTS)

On a host where Supertonic's ONNX graph runs on CPU, the defaults waste most of
a turn: the LLM finishes three sentences before TTS starts, and Supertonic's
`synthesize()` is blocking, so the first audio waits for the whole chunk.

```bash
make router-low        # same as `make router`, plus the flags below
make router-low 9R_MODEL=9r/routers9/deepseek-v4.1-flash-1017
```

`-LowLatency` sets `--stream_batch_sentences 1` (TTS starts on the first
sentence and overlaps the LLM), `--min_silence_ms 400`, and a short-reply system
prompt — the upstream default asks for "less than 20 words", which hosted
models routinely ignore. Individual knobs are `-BatchSentences`,
`-MinSilenceMs`, and `-Prompt`.

Measured on a 2 GB MX150 laptop (`tts_ttfa` from the pipeline's own turn log):

| | default | `-LowLatency` |
|---|---|---|
| TTS first audio | 9.4 s | **0.7 to 1.2 s** |
| pipeline e2e (warm) | 16 to 23 s | **7 to 8 s** |

STT still dominates the rest: Qwen Q5_K_M takes 2 to 3 s per pass on that card,
and the pipeline runs it three times per turn (two progressive, one final).


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

#### Per-session voice

`session.update` with `audio.output.voice` changes the voice for the rest of
that session; `F1`-`F5` female, `M1`-`M5` male. `ui/index.html` has a picker
that defers the change until the current reply finishes, since switching
mid-reply would splice two voices into one answer.

Upstream's `SupertonicTTSHandler` did not read it at all — it acknowledged the
update and kept the startup voice. `kokoro_handler.py` already resolves the
voice per turn; the local patch mirrors that. The voice is frozen per response:
with sentence-by-sentence TTS a `session.update` landing mid-reply would
otherwise change voice between sentences, so the first sentence decides and the
switch applies to the next reply.

`patches/0001-per-session-voice-and-ui-route.patch` holds the change (it also
adds the `/` UI route). The upstream clone is gitignored, so the patch does not
survive a fresh clone or a `git pull` that touches those files:

```bash
make patch      # re-apply it to speech-to-speech/ (idempotent)
```

If the voice stops changing while `session.updated` still arrives, that patch
is the first thing to check.

#### Serving it online

Locally nothing changes: `make router-low` binds `127.0.0.1:8765` and the page
at <http://127.0.0.1:8765/> talks to `ws://127.0.0.1:8765/v1/realtime`. A tunnel
is additive; without one, the server is reachable only from this machine.

```bash
make router-low   # text proxy stays off: a tunnel must not expose free chat
make tunnel       # cloudflared tunnel -> localhost:8765 (CF_TUNNEL=<name>)
```

The pipeline serves `ui/index.html` at `/` when `SPEECH_UI_DIR` is set
(`run.ps1` sets it in serve mode), so one hostname covers both the page and the
socket. The page picks `wss://<host>/v1/realtime` when loaded over HTTPS and
`ws://<host>:8765/v1/realtime` otherwise — a browser refuses a `ws://` socket
from an HTTPS page, and the tunnel only publishes one port.

#### Callers from other frontends

The backend is meant to be used by a frontend of your choosing — this repo's
page, one deployed to Cloudflare Pages, or a native app. Nothing about a
session is tied to the page, so put the login wherever the frontend is and let
the backend verify a shared token.

`run.ps1` generates one on first serve mode and keeps it in `_ws_token`
(gitignored), printing it at startup:

```text
Token: EXAMPLE-token-shown-at-startup-NOT-this-one
```

Pass it as `-WsToken` or `$env:SPEECH_WS_TOKEN` to pin one yourself. A caller
presents it in whichever way its client allows:

| Caller | How |
|---|---|
| Browser | `new WebSocket(url, ["realtime", "token.<T>"])` — a browser cannot set headers |
| `curl` / Python / Go | `Authorization: Bearer <T>`, or `?token=<T>` |
| This page | `?server=wss://host&token=<T>` (remembered in `localStorage`) |

`/v1/realtime/capabilities` stays open so a frontend can list languages and
voices before it has anything else; `/v1/pool`, `/v1/usage` and the socket are
gated. A wrong token closes with `1008` **before** a pipeline slot is claimed,
so a misconfigured frontend cannot lock a real user out. With no token
configured at all the server does not check — fine on `127.0.0.1`, not behind a
tunnel.

This is a shared secret, not per-person identity: anyone holding it is in,
revoking means changing the token and redeploying the frontend. That is a
deliberate trade for "easy to connect from any frontend" — see
`docs/networking.md` for the hostname and the deployment choices.

`make router-low` deliberately omits `--enable_llm_proxy`. With it on,
`/v1/chat/completions` answers 200 and anyone past the gate could spend the
router's credit on arbitrary text without sending audio. `make router` keeps it
on for local use.

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

| | Qwen bf16 (HF) | **Qwen Q5_K_M (GGUF)** | Qwen Q4_K_M | faster-whisper medium-id |
|---|---|---|---|---|
| File size | 1.57 GB | **645 MB** | 590 MB | ~0.8 GB |
| Corpus WER | 2.4 % | **2.4 %** | 4.8 % | 2.4 % |
| 6.5 to 10.9 s clip (Vulkan) | 728 ms | **142 ms** | 142 ms | 309 ms |
| Model load | ~2 s | **0.62 s** | 0.55 s | — |
| Peak VRAM | 1.67 GB | **~0.86 GB** | ~0.80 GB | ~0.8 GB (int8) |
| Code-switching | yes | **yes** | yes | no, forces Indonesian |

bf16 latency was measured through Transformers; the GGUF numbers through
transcribe.cpp on Vulkan, so the 5x gap is mostly the runtime, not the quant.
VRAM for bf16 is PyTorch's `max_memory_allocated`; GGUF figures are `nvidia-smi`
deltas, so treat the two columns as close, not exact.

Every other level on the same 8 clips, plus Whisper Turbo for comparison:

| Model | Size | Load | Median | WER | VRAM delta |
|---|---|---|---|---|---|
| Qwen Q8_0 | 850 MB | 0.86 s | 146 ms | 2.4 % | ~1.10 GB |
| Qwen Q6_K | 690 MB | 0.71 s | 144 ms | 2.4 % | ~0.95 GB |
| **Qwen Q5_K_M** | **645 MB** | 0.62 s | **142 ms** | **2.4 %** | **~0.86 GB** |
| Qwen Q4_K_M | 590 MB | 0.55 s | 142 ms | 4.8 % | ~0.80 GB |
| Whisper Turbo Q8_0 | 886 MB | 0.63 s | 157 ms | 10.7 % | ~0.92 GB |
| Whisper Turbo Q4_K_M | 536 MB | 0.40 s | 145 ms | 10.7 % | ~0.58 GB |

The Whisper comparison is subset-limited: 8 clips, and Whisper was not given a
language hint. The published FLEURS Indonesian numbers are Whisper Turbo 7.2 %
against Qwen3-ASR 8.49 %, so this table is **not** a general ranking — on this
subset the words Whisper misses are the same 8 clips Qwen gets right.

The WER tie between bf16 and Q5 is also 8 clips with identical errors, not a
real ranking, but it is enough to say quantisation down to Q5 costs nothing
measurable here. Q4 is where it breaks.

```bash
python bench/fetch_clips.py                          # needs the pipeline venv
python bench/bench_stt.py                            # qwen|fw|both
python bench/bench_gguf_stt.py                       # all GGUF levels + Whisper
python bench/bench_gguf_stt.py qwen-q5 whisper-q8    # or pick models
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

Supertonic runs on CPU via ONNX. The default graphs are a **q4** build of the
same checkpoint (127 MB instead of 380 MB): RTF 0.113 against 0.199, roughly
1.8x faster, and word-identical when round-tripped back through Qwen3-ASR
(5 Indonesian sentences, 2 runs each, 0 % WER both). `run.ps1` points
`SUPERTONIC_CACHE_DIR` at `models/supertonic-3-q4`, so nothing is patched.

```bash
make setup-q4    # fetch the graphs if they are missing (~130 MB)
```

Two community variants were rejected: fp16 gave no speedup on CPU (0.194), and
int8 produced noise (100 % WER, 7x slower). To compare them yourself, `python
bench/tts_listen.py "teks" F1` renders fp32 and q4 side by side.

Note that ASR-as-judge only catches gross breakage — it cannot tell timbre
apart, so quantisation artifacts still need a listen.

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
- **VRAM:** LLM ~3.1 GB + STT ~0.9 GB (Q5 GGUF, shared across slots) + VAD/TTS
  overhead fits in 16 GB with room for 3 pipeline slots. Each extra slot loads
  its own TTS, but the GGUF STT is one server process. `make hf` puts the old
  1.7 GB back.

## Troubleshooting

- **`Missing ...Qwen3-ASR-0.6B-Q5_K_M.gguf`** — run `make setup-gguf` (add
  `STT_QUANT=Q8_0` for another level; the path in the error names the file).
- **`GGUF STT server failed to start`** — run the server command from
  [STT: Qwen3-ASR as a GGUF](#stt-qwen3-asr-as-a-gguf) in a visible shell; the
  import error is usually `transcribe_cpp` missing from the venv
  (`make setup-gguf` reinstalls it).
- **Backend is `CPU` in the server's `READY` line** — Vulkan did not find a
  device. `vulkaninfo --summary` should list your GPU; update the driver, or
  force `--backend cpu` (slower, still works).
- **Port 8011 busy** — something already serves STT there and `run.ps1`
  reuses it. Stop it (`Get-Process python | ...`) or change `$sttPort` in
  `run.ps1`; the pipeline port is separate.
- **`All session slots are in use`** — see the first caveat; it clears in ~1.4 s.

## License

The wrapper and benchmarks here are MIT. Upstream `speech-to-speech` is MIT;
individual model checkpoints carry their own licences — check them before
commercial use (`facebook/mms-tts-ind`, for instance, is CC-BY-NC 4.0).
