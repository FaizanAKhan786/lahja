"""Production serving API.

Backends (auto-selected):
- LAHJA_MOCK_MODEL=1      → deterministic mock (CI / local dev, no GPU)
- LAHJA_GGUF_PATH=...     → llama-cpp-python (CPU/GPU, quantized GGUF)
- LAHJA_MODEL_DIR=...     → HF transformers merged model (GPU)

Run:
    uvicorn lahja.serve.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from lahja.serve.schemas import GenerateRequest, GenerateResponse, InfoResponse

log = logging.getLogger("lahja.serve")
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

_BACKEND: dict = {"name": None, "generate": None, "stream": None}


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _init_mock() -> None:
    def generate(prompt: str, max_new_tokens: int, temperature: float) -> str:
        return f"[mock·lahja] echo({len(prompt)} chars): {prompt[:80]}"

    def stream(prompt: str, max_new_tokens: int, temperature: float) -> Iterator[str]:
        for tok in generate(prompt, max_new_tokens, temperature).split(" "):
            yield tok + " "

    _BACKEND.update(name="mock", generate=generate, stream=stream)


def _init_gguf(path: str) -> None:
    from llama_cpp import Llama

    llm = Llama(model_path=path, n_ctx=4096, n_gpu_layers=-1, verbose=False)

    def generate(prompt: str, max_new_tokens: int, temperature: float) -> str:
        out = llm(prompt, max_tokens=max_new_tokens, temperature=temperature)
        return out["choices"][0]["text"]

    def stream(prompt: str, max_new_tokens: int, temperature: float) -> Iterator[str]:
        for chunk in llm(prompt, max_tokens=max_new_tokens,
                         temperature=temperature, stream=True):
            yield chunk["choices"][0]["text"]

    _BACKEND.update(name=f"gguf:{os.path.basename(path)}", generate=generate, stream=stream)


def _init_hf(model_dir: str) -> None:
    from threading import Thread

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    def _gen_kwargs(prompt, max_new_tokens, temperature):
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        return enc, dict(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.eos_token_id,
        )

    def generate(prompt: str, max_new_tokens: int, temperature: float) -> str:
        import torch as _t
        enc, kwargs = _gen_kwargs(prompt, max_new_tokens, temperature)
        with _t.no_grad():
            out = model.generate(**kwargs)
        return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def stream(prompt: str, max_new_tokens: int, temperature: float) -> Iterator[str]:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        _, kwargs = _gen_kwargs(prompt, max_new_tokens, temperature)
        Thread(target=model.generate, kwargs={**kwargs, "streamer": streamer}).start()
        yield from streamer

    _BACKEND.update(name=f"hf:{model_dir}", generate=generate, stream=stream)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("LAHJA_MOCK_MODEL"):
        _init_mock()
    elif os.getenv("LAHJA_GGUF_PATH"):
        _init_gguf(os.environ["LAHJA_GGUF_PATH"])
    elif os.getenv("LAHJA_MODEL_DIR"):
        _init_hf(os.environ["LAHJA_MODEL_DIR"])
    else:
        log.warning("No model env set — falling back to mock backend")
        _init_mock()
    log.info("Backend ready: %s", _BACKEND["name"])
    yield


app = FastAPI(title="Lahja — Gulf Arabic Dialect LLM", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": _BACKEND["name"]}


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    return InfoResponse(backend=_BACKEND["name"] or "uninitialized", version=app.version)


@app.post("/generate")
def generate(req: GenerateRequest):
    if _BACKEND["generate"] is None:
        raise HTTPException(503, "Model not loaded")

    if req.stream:
        def sse() -> AsyncIterator[str]:
            for token in _BACKEND["stream"](req.prompt, req.max_new_tokens, req.temperature):
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    start = time.perf_counter()
    text = _BACKEND["generate"](req.prompt, req.max_new_tokens, req.temperature)
    return GenerateResponse(
        text=text,
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
        backend=_BACKEND["name"],
    )
