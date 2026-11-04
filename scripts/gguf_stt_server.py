"""OpenAI-compatible STT server backed by transcribe.cpp and a Qwen3-ASR GGUF.

The speech-to-speech pipeline speaks the OpenAI transcription protocol, so a
small HTTP wrapper is enough to swap in a quantized GGUF without patching the
upstream clone. run.ps1 starts this on 127.0.0.1:8011 and points --stt openai
at it.

    python scripts/gguf_stt_server.py --model models/qwen-asr-gguf/Qwen3-ASR-0.6B-Q5_K_M.gguf
"""
import argparse
import io
import threading
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from transcribe_cpp import Model

SAMPLE_RATE = 16000
_lock = threading.Lock()


def decode_wav(data: bytes) -> np.ndarray:
    """Decode any WAV to mono 16 kHz float32; the pipeline already sends 16 kHz."""
    try:
        audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid WAV: {exc}") from exc
    audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        count = int(round(len(audio) * SAMPLE_RATE / sample_rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, count), np.arange(len(audio)), audio)
    return np.ascontiguousarray(audio, dtype=np.float32)


def create_app(model_path: str, backend: str = "auto") -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model = Model(model_path, backend=backend)
        state["model"] = model
        state["session"] = model.session()
        print(
            f"READY {model.arch}/{model.variant} backend={model.backend} file={model_path}",
            flush=True,
        )
        try:
            yield
        finally:
            state["session"].close()
            model.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "backend": state["model"].backend}

    @app.post("/v1/audio/transcriptions")
    def transcribe(
        file: UploadFile = File(...),
        language: str | None = Form(None),
        response_format: str = Form("json"),
        model: str | None = Form(None),  # accepted and ignored, as the pipeline sends one
    ) -> object:
        pcm = decode_wav(file.file.read())
        with _lock:
            result = state["session"].run(pcm, language=language or None, timestamps="none")
        if response_format == "text":
            return PlainTextResponse(result.text)
        detected = result.language or language
        # An unforced language means the model detected it. The pipeline's
        # "-auto" suffix marks that: Supertonic and the LLM strip it again.
        if language is None and detected:
            detected = f"{detected}-auto"
        return {"text": result.text, "language": detected}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="auto", help="auto, vulkan, cuda or cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model, args.backend), host=args.host, port=args.port, log_level="warning")
