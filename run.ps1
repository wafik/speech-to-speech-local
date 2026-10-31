# RTX 5060 Ti 16GB - speech-to-speech launcher
# Stack: Qwen3-ASR (STT) -> gemma4:e4b @ Ollama (LLM) -> Supertonic (TTS)
# Reasoning OFF is the whole trick: thinking models burn ~1K reasoning tokens/turn
# otherwise (qwen3.5: 14s -> 0.6s, gemma4: 2.1s -> 0.3s).
#
# Usage:  powershell -ExecutionPolicy Bypass -File run.ps1
#         powershell -ExecutionPolicy Bypass -File run.ps1 -English
#         powershell -ExecutionPolicy Bypass -File run.ps1 -Model qwen3.5:9b
#         powershell -ExecutionPolicy Bypass -File run.ps1 -Mode serve
#         powershell -ExecutionPolicy Bypass -File run.ps1 -Online
#
# NOTE: keep this file ASCII-only. PowerShell 5.1 reads .ps1 as cp1252 without a
# BOM, so an em-dash inside a string turns into a smart quote and breaks parsing.

param(
    [string]$Model = "gemma4:e4b",
    [string]$Tts = "supertonic",
    [string]$Stt = "qwen3-asr",
    [double]$Speed = 1.0,
    [string]$Lang = "",       # pin Supertonic language; empty = follow the detected one
    [string]$HfHome = "",     # model cache dir; defaults to $env:HF_HOME
    [string]$Mode = "local",  # local | serve | talk
    [int]$Pipelines = 0,      # serve only; 0 = auto (3). Each slot holds its own STT+TTS.
    [switch]$English,         # parakeet + kokoro: lower latency, English only
    [switch]$Online           # omit to stay fully offline once models are cached
)

# Default is the multilingual pair: Qwen3-ASR auto-detects the language of every
# turn, and Supertonic speaks 32 of them. Parakeet has 25 European languages and
# Kokoro 8 voices, so neither can speak Indonesian at all.
if ($English) { $Stt = "parakeet-tdt"; $Tts = "kokoro" }

$ErrorActionPreference = "Stop"
# Layout: this script sits next to the cloned upstream repo.
#   <repo>/run.ps1
#   <repo>/speech-to-speech/        <- git clone huggingface/speech-to-speech
$Root = Join-Path $PSScriptRoot "speech-to-speech"
if (-not (Test-Path "$Root\.venv\Scripts\speech-to-speech.exe")) {
    Write-Host "Missing $Root\.venv - run 'make setup' first." -ForegroundColor Red
    exit 1
}
# Model cache. HF_HOME is inherited if already set; otherwise fall back to the
# user default. All checkpoints live under $HF_HOME\hub, so one variable moves
# them all - set it once with: setx HF_HOME D:\your\cache
if ($HfHome) { $env:HF_HOME = $HfHome }
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $env:USERPROFILE ".cache\huggingface" }
$env:VIRTUAL_ENV = "$Root\.venv"
if (-not $Online) {
    # All checkpoints live under $env:HF_HOME\hub; skip every HF Hub round-trip.
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:HF_HUB_DISABLE_TELEMETRY = "1"
}

# Silent TCP probe - Test-NetConnection writes a spinner into the console output.
function Test-Port([int]$Port) {
    $c = New-Object Net.Sockets.TcpClient
    try { $c.Connect("127.0.0.1", $Port); $true } catch { $false } finally { $c.Dispose() }
}

# Ollama server: start if not already listening on 11434
if (-not (Test-Port 11434)) {
    Write-Host "Starting Ollama..." -ForegroundColor Yellow
    Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
    do { Start-Sleep 1 } until (Test-Port 11434)
}

# Port 8765 is held by any previous session; a second one fails to bind.
if (Test-Port 8765) {
    Write-Host "Port 8765 busy - stopping the previous session..." -ForegroundColor Yellow
    Get-Process -Name speech-to-speech -ErrorAction SilentlyContinue | Stop-Process -Force
    do { Start-Sleep 1 } until (-not (Test-Port 8765))
}

# A slot stays busy ~1.4s after a client closes, so a reconnect inside that
# window is rejected with "All session slots are in use". Extra slots absorb it.
$slots = 0
if ($Mode -eq "serve") {
    $slots = if ($Pipelines -gt 0) { $Pipelines } else { 3 }
}

Write-Host "Model: $Model | STT: $Stt | TTS: $Tts | offline: $(-not $Online)" -ForegroundColor Cyan
if ($Mode -eq "serve") { Write-Host "Slots: $slots | ws://127.0.0.1:8765/v1/realtime" -ForegroundColor Cyan }
Write-Host "Speak, pause, listen. Ctrl+C to quit." -ForegroundColor Cyan

Set-Location $Root
$cliArgs = @(
    $Mode,
    "--llm_backend", "chat-completions",
    "--model_name", $Model,
    "--responses_api_base_url", "http://127.0.0.1:11434/v1",
    "--responses_api_api_key", "ollama",
    "--responses_api_reasoning_effort", "none",
    "--stt", $Stt,
    "--tts", $Tts
)
if ($Tts -eq "kokoro") { $cliArgs += @("--kokoro_speed", $Speed) }
# Left unset, Supertonic follows the language Qwen3-ASR detected on each turn.
if ($Lang) { $cliArgs += @("--supertonic_tts_lang", $Lang) }
# local mode owns the mic; serve mode is headless for other agents
if ($Mode -eq "local") { $cliArgs += "--local_audio_block_mic_during_playback" }
# serve mode also exposes /v1/chat/completions so text-only agents can share the same LLM
if ($Mode -eq "serve") {
    $cliArgs += "--enable_llm_proxy"
    $cliArgs += @("--num_pipelines", $slots)
}

& .\.venv\Scripts\speech-to-speech.exe @cliArgs
