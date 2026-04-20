"""
Probe Inference Service
=========================
FastAPI server that loads the base model + artifacts/probe_config.json
and scores prompts for harmfulness using the refusal-direction probe.

Optimisations in this build:
  * model truncated to best_layer+1 layers (no wasted compute past the probe)
  * lm_head dropped; forward hook captures the hidden state we need
  * output_hidden_states=True is NOT used (no materialising all layers)

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
from contextlib import asynccontextmanager
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

STATE = {
    "model": None,
    "tokenizer": None,
    "probe": None,
    "global_dir": None,
    "hook_buffer": None,
    "ready": False,
}


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=1)
    threshold: Optional[float] = None


class BatchScoreRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=64)
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


def _load_model_and_probe():
    if not PROBE_CONFIG.exists():
        raise RuntimeError(f"probe config not found: {PROBE_CONFIG}")

    log.info(f"loading probe config {PROBE_CONFIG}")
    probe = json.loads(PROBE_CONFIG.read_text())
    model_id = probe.get("model", MODEL_PATH)
    layer = int(probe["best_layer"])
    log.info(f"model={model_id}  layer={layer}  "
             f"threshold={probe['threshold_block']}  device={DEVICE}")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=DEVICE,
    )
    model.eval()

    # Truncate: hidden_states[layer] in HF equals the output of the first `layer`
    # transformer blocks (hidden_states[0] is embeddings). Keep exactly that many.
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        n_total = len(base.layers)
        if layer > n_total:
            raise RuntimeError(
                f"best_layer={layer} > model depth {n_total}. "
                f"Regenerate probe_config.json against this model."
            )
        base.layers = base.layers[:layer]
        log.info(f"truncated transformer layers: {n_total} -> {layer}")
    else:
        log.warning("base.layers not found; running full forward")

    # Drop the LM head so it's never called
    if hasattr(model, "lm_head"):
        model.lm_head = torch.nn.Identity()

    # Forward hook on the last (kept) layer grabs its output.
    hook_buf: dict = {}

    def _hook(_mod, _inp, out):
        # layer forward returns either Tensor or (Tensor, ...) depending on model
        hook_buf["h"] = out[0] if isinstance(out, tuple) else out

    base.layers[-1].register_forward_hook(_hook)

    direction = torch.tensor(probe["global_direction"],
                             dtype=torch.float32, device=DEVICE)

    STATE["model"]       = model
    STATE["base"]        = base
    STATE["tokenizer"]   = tok
    STATE["probe"]       = probe
    STATE["global_dir"]  = direction
    STATE["hook_buffer"] = hook_buf
    STATE["ready"]       = True
    log.info("service ready")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model_and_probe()
    yield


app = FastAPI(title="Probe Inference Service", version="1.1", lifespan=lifespan)


@torch.inference_mode()
def _score_texts(texts: List[str]) -> List[float]:
    tok  = STATE["tokenizer"]
    base = STATE["base"]
    d    = STATE["global_dir"]
    buf  = STATE["hook_buffer"]

    enc = tok(texts, return_tensors="pt", truncation=True,
              max_length=MAX_LENGTH, padding=True).to(DEVICE)

    # Run the base transformer (no lm_head). The hook populates buf["h"].
    base(**enc)
    hidden = buf["h"].float()                  # [B, T, H]
    mask   = enc["attention_mask"]              # [B, T]
    last   = mask.sum(dim=1) - 1                # last real token index per row
    idx    = torch.arange(hidden.size(0), device=hidden.device)
    last_hidden = hidden[idx, last, :]          # [B, H]
    return (last_hidden @ d).tolist()


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
    # CLI flags win over env vars; propagate back so lifespan startup sees them
    os.environ["MODEL"]        = args.model
    os.environ["PROBE_CONFIG"] = args.probe_config
    os.environ["MAX_LENGTH"]   = str(args.max_length)
    MODEL_PATH   = args.model
    PROBE_CONFIG = Path(args.probe_config)
    MAX_LENGTH   = args.max_length
    log.info(f"starting uvicorn on {args.host}:{args.port}")
    uvicorn.run("server:app", host=args.host, port=args.port,
                log_level=args.log_level)
