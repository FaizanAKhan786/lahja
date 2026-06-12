# Lahja (لهجة) — Gulf Arabic Dialect LLM Pipeline

[![CI](https://github.com/faaizan786/lahja/actions/workflows/ci.yml/badge.svg)](https://github.com/faaizan786/lahja/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

**End-to-end pipeline for adapting open LLMs to Gulf Arabic dialects** (Qatari, Emirati, Saudi, Kuwaiti):
QLoRA fine-tuning → evaluation harness → INT4/GGUF quantization → production FastAPI serving with streaming.

Built to run **fully offline** after model download — no external API calls — reflecting real-world
air-gapped enterprise and government deployment constraints.

```
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │  Data Prep    │──▶│ QLoRA Train  │──▶│  Evaluate    │──▶│  Quantize     │
 │ (instruction  │   │ (4-bit base, │   │ (dialect F1, │   │ (merge LoRA → │
 │  formatting,  │   │  PEFT/LoRA   │   │  ppl, judge) │   │  INT4 / GGUF) │
 │  chat template)│  │  adapters)   │   │              │   │               │
 └──────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                                  ▼
                                                       ┌────────────────────┐
                                                       │  FastAPI Serving    │
                                                       │  /generate (stream) │
                                                       │  /health  /info     │
                                                       └────────────────────┘
```

## Why this exists

Modern Standard Arabic (MSA) models underperform badly on Gulf dialects — the Arabic actually
spoken and typed in Qatar, the UAE, Saudi Arabia, and Kuwait. This repo is a reproducible,
config-driven pipeline for closing that gap with parameter-efficient fine-tuning on a single GPU.

## Features

- **QLoRA fine-tuning** — 4-bit NF4 base model + LoRA adapters via PEFT; trains an 8B model on a single 24 GB GPU
- **Config-driven** — every run defined by a YAML config; no code edits to change model, data, or hyperparameters
- **Evaluation harness** — dialect identification accuracy/macro-F1, held-out perplexity, and optional LLM-as-judge scoring
- **Quantization** — LoRA merge + INT4 export, with a GGUF conversion script for llama.cpp/Ollama deployment
- **Production serving** — FastAPI with token streaming (SSE), request validation, health/readiness probes, and a mock-model mode for CI
- **MLOps hygiene** — unit tests, GitHub Actions CI, Dockerfile, structured logging, reproducible seeds

## Quickstart

```bash
git clone https://github.com/faaizan786/lahja && cd lahja
pip install -e ".[dev]"

# 1. Smoke-run the full pipeline on the bundled sample data (CPU-friendly, tiny model)
python -m lahja.data.prepare --config configs/sample.yaml
python -m lahja.train.qlora --config configs/sample.yaml --max-steps 5

# 2. Serve (mock mode — no GPU needed, used by CI)
LAHJA_MOCK_MODEL=1 uvicorn lahja.serve.api:app --port 8000
curl -X POST localhost:8000/generate -H 'Content-Type: application/json' \
     -d '{"prompt": "شلونك؟ وش أخبارك", "max_new_tokens": 64}'
```

### Full training run (GPU)

```bash
# Edit configs/training.yaml: base model, dataset, LoRA rank, etc.
python -m lahja.data.prepare --config configs/training.yaml
python -m lahja.train.qlora   --config configs/training.yaml
python -m lahja.train.merge   --config configs/training.yaml   # merge adapters
python -m lahja.eval.evaluate --config configs/training.yaml   # metrics → reports/
bash scripts/quantize_gguf.sh outputs/merged                    # GGUF for llama.cpp/Ollama
uvicorn lahja.serve.api:app --port 8000                         # serve merged model
```

## Data

The repo ships with a small **bundled sample dataset** (`data/sample/`) so the pipeline runs
end-to-end out of the box. For real training, plug in any instruction-format JSONL
(`{"instruction": ..., "input": ..., "output": ..., "dialect": ...}`) — public options include:

| Dataset | Dialects | Access |
|---|---|---|
| NADI shared task | 18 country-level dialects | Registration (free, academic) |
| MADAR corpus | 25 city-level dialects | Registration (free, academic) |
| QADI (QCRI) | 18 country-level dialects | Public (Twitter IDs) |

Point `data.train_file` in your config at the prepared JSONL. The prep module handles
chat-template formatting, deduplication, and train/eval splitting.

## Evaluation

`python -m lahja.eval.evaluate` produces a JSON + Markdown report:

- **Dialect ID** — accuracy and macro-F1 on a held-out labelled split
- **Perplexity** — on held-out dialect text, base vs fine-tuned
- **LLM-as-judge** (optional) — pairwise fluency/dialect-fidelity wins vs the base model, judged by a local model via Ollama (keeps the loop fully offline)

Results land in `reports/` so every run is auditable.

## Serving API

| Endpoint | Description |
|---|---|
| `POST /generate` | Generate (optionally stream via `"stream": true` — SSE) |
| `GET /health` | Liveness probe |
| `GET /info` | Model, adapter, quantization, and build metadata |

Backends: merged HF model (GPU), GGUF via llama-cpp-python (CPU/GPU), or mock (CI).
Selected automatically from config/env — see `configs/serving.yaml`.

## Docker

```bash
docker build -t lahja .
docker run -p 8000:8000 -e LAHJA_MOCK_MODEL=1 lahja          # mock
docker run --gpus all -p 8000:8000 -v ./outputs:/models lahja # real model
```

## Project layout

```
src/lahja/
  data/prepare.py     # JSONL → formatted train/eval splits
  data/format.py      # chat-template & instruction formatting (pure python, unit-tested)
  train/qlora.py      # 4-bit QLoRA training loop (PEFT + TRL)
  train/merge.py      # adapter merge → standalone model
  eval/evaluate.py    # dialect F1, perplexity, LLM-as-judge
  serve/api.py        # FastAPI app (stream, health, mock mode)
  serve/schemas.py    # pydantic request/response models
configs/              # YAML run configs (sample + full training + serving)
scripts/              # train / quantize / serve helpers
tests/                # unit tests (run without GPU)
```

## License

MIT — see [LICENSE](LICENSE).
