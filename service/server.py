"""
Probe Inference Service
=========================
FastAPI server that loads the base model + artifacts/probe_config.json
and scores prompts for harmfulness using the refusal-direction probe.

Boot (env vars):
  MODEL=votal-ai/vai35-4B \
  PROBE_CONFIG=./artifacts/probe_config.json \
  PORT=8000 \
  python service/server.py

Boot (CLI flags, override env):
  python service/server.py --port 9000 --host 0.0.0.0 \
      --probe-config ./artifacts/probe_config.json

Endpoints:
  GET  /health            -> liveness
  GET  /info              -> probe metadata (layer, threshold, calibration)
  POST /score             -> {"text": "..."} -> score + block decision
  POST /score/batch       -> {"texts": [...]} -> batched scoring
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("probe")

DEFAULT_PORT = 8000

MODEL_PATH   = os.environ.get("MODEL", "votal-ai/vai35-4B")
PROBE_CONFIG = Path(os.environ.get("PROBE_CONFIG", "./artifacts/probe_config.json"))
MAX_LENGTH   = int(os.environ.get("MAX_LENGTH", "512"))
PORT         = int(os.environ.get("PORT", str(DEFAULT_PORT)))
HOST         = os.environ.get("HOST", "0.0.0.0")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Probe Inference Service", version="1.0")

STATE = {
    "model": None,
    "tokenizer": None,
    "probe": None,
    "global_dir": None,
    "ready": False,
}


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=1)
    threshold: Optional[float] = None


class BatchScoreRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=64)
    threshold: Optional[float] = None


class ScoreResponse(BaseModel):
    score: float
    blocked: bool
    threshold: float
    latency_ms: float


class BatchScoreResponse(BaseModel):
    scores: List[float]
    blocked: List[bool]
    threshold: float
    latency_ms: float


@app.on_event("startup")
def load_everything():
    if not PROBE_CONFIG.exists():
        raise RuntimeError(f"probe config not found: {PROBE_CONFIG}")

    log.info(f"loading probe config {PROBE_CONFIG}")
    probe = json.loads(PROBE_CONFIG.read_text())
    model_id = probe.get("model", MODEL_PATH)
    log.info(f"model={model_id}  layer={probe['best_layer']}  "
             f"threshold={probe['threshold_block']}  device={DEVICE}")

    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=DEVICE,
        output_hidden_states=True,
    )
    model.eval()

    direction = torch.tensor(probe["global_direction"], dtype=torch.float32, device=DEVICE)

    STATE["model"]      = model
    STATE["tokenizer"]  = tok
    STATE["probe"]      = probe
    STATE["global_dir"] = direction
    STATE["ready"]      = True
    log.info("service ready")


@torch.inference_mode()
def _score_texts(texts: List[str]) -> List[float]:
    tok   = STATE["tokenizer"]
    model = STATE["model"]
    probe = STATE["probe"]
    d     = STATE["global_dir"]
    layer = probe["best_layer"]

    enc = tok(texts, return_tensors="pt", truncation=True,
              max_length=MAX_LENGTH, padding=True).to(DEVICE)
    out = model(**enc, output_hidden_states=True)

    hidden = out.hidden_states[layer].float()           # [B, T, H]
    mask   = enc["attention_mask"]                      # [B, T]
    last   = mask.sum(dim=1) - 1                        # index of last real token
    idx    = torch.arange(hidden.size(0), device=hidden.device)
    last_hidden = hidden[idx, last, :]                  # [B, H]
    scores = (last_hidden @ d).tolist()
    return scores


@app.get("/health")
def health():
    return {"ok": STATE["ready"], "device": DEVICE}


@app.get("/info")
def info():
    if not STATE["ready"]:
        raise HTTPException(503, "model not loaded")
    p = STATE["probe"]
    return {
        "model":       p.get("model"),
        "version":     p.get("version"),
        "best_layer":  p["best_layer"],
        "hidden_size": p.get("hidden_size"),
        "threshold_block": p["threshold_block"],
        "threshold_allow": p.get("threshold_allow"),
        "calibrated":  p.get("calibrated", False),
        "calibration": p.get("calibration", {}),
        "categories":  sorted(p.get("categories", {}).keys()),
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    if not STATE["ready"]:
        raise HTTPException(503, "model not loaded")
    t0 = time.perf_counter()
    s  = _score_texts([req.text])[0]
    th = req.threshold if req.threshold is not None else STATE["probe"]["threshold_block"]
    return ScoreResponse(
        score=s, blocked=s >= th, threshold=th,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(req: BatchScoreRequest):
    if not STATE["ready"]:
        raise HTTPException(503, "model not loaded")
    t0 = time.perf_counter()
    scores = _score_texts(req.texts)
    th = req.threshold if req.threshold is not None else STATE["probe"]["threshold_block"]
    return BatchScoreResponse(
        scores=scores, blocked=[s >= th for s in scores], threshold=th,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe inference service (FastAPI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default=HOST,
                   help="bind address (env: HOST)")
    p.add_argument("--port", type=int, default=PORT,
                   help="listen port (env: PORT)")
    p.add_argument("--probe-config", default=str(PROBE_CONFIG),
                   help="path to probe_config.json (env: PROBE_CONFIG)")
    p.add_argument("--model", default=MODEL_PATH,
                   help="fallback HF model id (env: MODEL)")
    p.add_argument("--max-length", type=int, default=MAX_LENGTH,
                   help="tokenizer truncation length (env: MAX_LENGTH)")
    p.add_argument("--log-level", default="info",
                   help="uvicorn log level")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # CLI flags win over env vars; propagate back so @on_event startup sees them
    os.environ["MODEL"]        = args.model
    os.environ["PROBE_CONFIG"] = args.probe_config
    os.environ["MAX_LENGTH"]   = str(args.max_length)
    MODEL_PATH   = args.model
    PROBE_CONFIG = Path(args.probe_config)
    MAX_LENGTH   = args.max_length
    log.info(f"starting uvicorn on {args.host}:{args.port}")
    uvicorn.run("server:app", host=args.host, port=args.port,
                log_level=args.log_level)
