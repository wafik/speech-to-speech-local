# RTX 5060 Ti 16GB - speech-to-speech launcher
# Stack: Qwen3-ASR Q5_K_M GGUF (STT, transcribe.cpp/Vulkan)
#        -> gemma4:e4b @ Ollama (LLM) -> Supertonic (TTS)
# Reasoning OFF is the whole trick: thinking models burn ~1K reasoning tokens/turn
# otherwise (qwen3.5: 14s -> 0.6s, gemma4: 2.1s -> 0.3s).
#
# Usage:  powershell -ExecutionPolicy Bypass -File run.ps1
#         powershell -ExecutionPolicy Bypass -File run.ps1 -English
#         powershell -ExecutionPolicy Bypass -File run.ps1 -Model qwen3.5:9b
#         powershell -ExecutionPolicy Bypass -File run.ps1 -Mode serve
#         powershell -ExecutionPolicy Bypass -File run.ps1 -Online
#         powershell -ExecutionPolicy Bypass -File run.ps1 -Stt qwen3-asr  (HF bf16)
#         powershell -ExecutionPolicy Bypass -File run.ps1 -SttQuant Q8_0
#
# NOTE: keep this file ASCII-only. PowerShell 5.1 reads .ps1 as cp1252 without a
# BOM, so an em-dash inside a string turns into a smart quote and breaks parsing.

param(
    [string]$Model = "gemma4:e4b",
    [string]$Tts = "supertonic",
    [string]$Stt = "gguf",     # gguf (Qwen3-ASR via transcribe.cpp) | qwen3-asr (HF bf16) | parakeet-tdt
    [string]$SttQuant = "Q5_K_M", # Q5_K_M (default) | Q6_K | Q8_0 | Q4_K_M
    [string]$SttModel = "",  # full path to a .gguf; overrides -SttQuant
    [double]$Speed = 1.0,
    [string]$Lang = "",       # pin Supertonic language; empty = follow the detected one
    [string]$Voice = "F1",    # Supertonic style: F1-F5 female, M1-M5 male
    [string]$HfHome = "",     # model cache dir; defaults to $env:HF_HOME
    [string]$ApiBase = "http://127.0.0.1:11434/v1",  # any OpenAI-compatible LLM endpoint
    [string]$ApiKey = "",     # empty: $env:9R_KEY, then the Claude Code config, then "none"
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
# them all - set it once with: setx HF_HOME E:\models\hf
if ($HfHome) { $env:HF_HOME = $HfHome }
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $env:USERPROFILE ".cache\huggingface" }

# A terminal opened before `setx HF_HOME` keeps the old value, and offline mode
# then fails 40 s in, deep inside transformers. Check up front: fall back to the
# persisted value when it has the checkpoint, otherwise stop immediately.
function Test-Cached([string]$CacheDir, [string]$Repo) {
    $snaps = Join-Path $CacheDir "hub\models--$($Repo -replace '/', '--')\snapshots"
    if (-not (Test-Path $snaps)) { return $false }
    (Get-ChildItem $snaps -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0
}
# GGUF default: the weights are a local file, so none of the HF cache checks
# apply. Only the HF backends need a cached repo.
$sttRepo = ""
if ($Stt -eq "parakeet-tdt") { $sttRepo = "nvidia/parakeet-tdt-0.6b-v3" }
elseif ($Stt -eq "qwen3-asr") { $sttRepo = "Qwen/Qwen3-ASR-0.6B-hf" }
if ($sttRepo) {
    if (-not (Test-Cached $env:HF_HOME $sttRepo)) {
        $saved = [Environment]::GetEnvironmentVariable("HF_HOME", "User")
        if ($saved -and $saved -ne $env:HF_HOME -and (Test-Cached $saved $sttRepo)) {
            Write-Host "HF_HOME=$env:HF_HOME has no $sttRepo - using $saved" -ForegroundColor Yellow
            Write-Host "  (this shell predates 'setx HF_HOME'; reopen it to drop this notice)" -ForegroundColor DarkGray
            $env:HF_HOME = $saved
        }
    }
    if (-not $Online -and -not (Test-Cached $env:HF_HOME $sttRepo)) {
        Write-Host "No $sttRepo under $env:HF_HOME\hub, and offline mode is on." -ForegroundColor Red
        Write-Host "  Run once with -Online, or pass -HfHome <cache that has it>." -ForegroundColor Red
        exit 1
    }
}
$env:VIRTUAL_ENV = "$Root\.venv"
# Supertonic resolves its model dir from this env var. The q4 graphs are half
# the size and ~1.8x faster than the fp32 default, word-identical in testing.
# Absolute path: the pipeline runs with cwd set to speech-to-speech/.
if ($Tts -eq "supertonic") {
    $q4 = Join-Path $PSScriptRoot "models\supertonic-3-q4"
    if (-not (Test-Path "$q4\onnx\vector_estimator.onnx")) {
        Write-Host "Missing $q4 - run 'make setup-q4' first." -ForegroundColor Red
        exit 1
    }
    $env:SUPERTONIC_CACHE_DIR = $q4
}

# GGUF STT weights are a local file: no HF cache, no offline-mode issue.
# Q5_K_M is the default - same 2.4% WER as bf16 on the Indonesian clips at
# 645 MB instead of 1.57 GB. Q4_K_M regressed to 4.8%, so it is not offered.
$sttPort = 8011
$gguf = ""
if ($Stt -eq "gguf") {
    $gguf = if ($SttModel) { $SttModel }
            else { Join-Path $PSScriptRoot "models\qwen-asr-gguf\Qwen3-ASR-0.6B-$SttQuant.gguf" }
    if (-not (Test-Path $gguf)) {
        Write-Host "Missing $gguf - run 'make setup-gguf' first." -ForegroundColor Red
        Write-Host "  Other quants: make setup-gguf STT_QUANT=$SttQuant" -ForegroundColor Red
        exit 1
    }
}
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

# Key for the LLM endpoint. Ollama ignores it; a hosted/router endpoint checks
# it. The local router takes the same token Claude Code already uses, and that
# one lives in the CLI config rather than the shell, so fall back to it. Read
# only - nothing is written anywhere.
if (-not $ApiKey -and $ApiBase -notlike "*:11434*") {
    $ApiKey = if ($env:9R_KEY) { $env:9R_KEY }
              elseif ($env:ANTHROPIC_AUTH_TOKEN -and $env:ANTHROPIC_BASE_URL -like "*:20128*") { $env:ANTHROPIC_AUTH_TOKEN }
              else {
                  $cfg = Join-Path $env:USERPROFILE ".claude\settings.json"
                  if (Test-Path $cfg) { (Get-Content $cfg -Raw | ConvertFrom-Json).env.ANTHROPIC_AUTH_TOKEN }
              }
}
if (-not $ApiKey) { $ApiKey = if ($ApiBase -like "*:11434*") { "ollama" } else { "none" } }

# STT sidecar: transcribe.cpp behind an OpenAI-compatible endpoint. One model
# instance serves every pipeline slot, so 3 slots cost one copy of the weights.
# If something already listens on 8011, reuse it and leave it running.
$sttServer = $null
if ($Stt -eq "gguf" -and -not (Test-Port $sttPort)) {
    Write-Host "Starting GGUF STT on 127.0.0.1:$sttPort ($(Split-Path $gguf -Leaf))..." -ForegroundColor Yellow
    $sttServer = Start-Process -FilePath "$Root\.venv\Scripts\python.exe" `
        -ArgumentList @("-I", (Join-Path $PSScriptRoot "scripts\gguf_stt_server.py"),
                        "--model", $gguf, "--port", $sttPort, "--backend", "auto") `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        if ($sttServer.HasExited) { break }
        try {
            $null = Invoke-RestMethod "http://127.0.0.1:$sttPort/health" -TimeoutSec 2
            $ready = $true; break
        } catch { Start-Sleep 1 }
    }
    if (-not $ready) {
        Write-Host "GGUF STT server failed to start (see the model path/backend)." -ForegroundColor Red
        if ($sttServer -and -not $sttServer.HasExited) { $sttServer.Kill() }
        exit 1
    }
}
elseif ($Stt -eq "gguf") {
    Write-Host "Reusing the GGUF STT already on 127.0.0.1:$sttPort" -ForegroundColor DarkGray
}

# Ollama server: start if not already listening on 11434. Only when the LLM
# actually lives there - a remote/9router model needs no local Ollama.
if ($ApiBase -like "*:11434*" -and -not (Test-Port 11434)) {
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

$sttLabel = if ($Stt -eq "gguf") { "gguf/$SttQuant" } else { $Stt }
Write-Host "Model: $Model | STT: $sttLabel | TTS: $Tts ($Voice) | offline: $(-not $Online)" -ForegroundColor Cyan
if ($Mode -eq "serve") { Write-Host "Slots: $slots | ws://127.0.0.1:8765/v1/realtime" -ForegroundColor Cyan }
Write-Host "Speak, pause, listen. Ctrl+C to quit." -ForegroundColor Cyan

Set-Location $Root
$sttBackend = if ($Stt -eq "gguf") { "openai" } else { $Stt }
$cliArgs = @(
    $Mode,
    "--llm_backend", "chat-completions",
    "--model_name", $Model,
    "--responses_api_base_url", $ApiBase,
    "--responses_api_api_key", $ApiKey,
    "--responses_api_reasoning_effort", "none",
    "--stt", $sttBackend,
    "--tts", $Tts
)
if ($Stt -eq "gguf") {
    $cliArgs += @("--openai_stt_base_url", "http://127.0.0.1:$sttPort/v1")
    # The server ignores the model id, but the handler refuses an empty
    # model+language pair, and both should stay unset for auto-detection.
    $cliArgs += @("--openai_stt_model", "qwen3-asr")
}
if ($Tts -eq "kokoro") { $cliArgs += @("--kokoro_speed", $Speed) }
# Left unset, Supertonic follows the language Qwen3-ASR detected on each turn.
if ($Lang) { $cliArgs += @("--supertonic_tts_lang", $Lang) }
if ($Tts -eq "supertonic") { $cliArgs += @("--supertonic_tts_voice", $Voice) }
# local mode owns the mic; serve mode is headless for other agents
if ($Mode -eq "local") { $cliArgs += "--local_audio_block_mic_during_playback" }
# serve mode also exposes /v1/chat/completions so text-only agents can share the same LLM
if ($Mode -eq "serve") {
    $cliArgs += "--enable_llm_proxy"
    $cliArgs += @("--num_pipelines", $slots)
}

try {
    & .\.venv\Scripts\speech-to-speech.exe @cliArgs
    $code = $LASTEXITCODE
}
finally {
    # Stop only the server this run started; a pre-existing one stays up.
    if ($sttServer -and -not $sttServer.HasExited) {
        $sttServer.Kill()
        $sttServer.WaitForExit(5000) | Out-Null
    }
}
exit $code