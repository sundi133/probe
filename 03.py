"""
STEP 3 — Build Probe Config
=============================
Reads:   artifacts/directions.pt
         artifacts/top_layers.json
         data/probe_pairs.json
Produces: artifacts/probe_config.json   (ready for inference)

Builds per-category directions + global probe config.
Default threshold is conservative; step4 will calibrate it.

Usage:
  python step3_build_probe.py
  MODEL=votal-ai/vai35-9B python step3_build_probe.py
"""

import os, json, gc
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.environ.get("MODEL", "votal-ai/vai35-4B")
DATA_DIR     = Path(os.environ.get("DATA_DIR",  "./data"))
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "./artifacts"))
MAX_LENGTH   = 512

# ── Load ──────────────────────────────────────────────────────────────────────
directions = torch.load(ARTIFACT_DIR / "directions.pt")

with open(ARTIFACT_DIR / "top_layers.json") as f:
    top_info = json.load(f)

with open(DATA_DIR / "probe_pairs.json") as f:
    pairs = json.load(f)

best_layer  = top_info["best_layer"]
top_layers  = top_info["top_layers"][:8]   # use top 8 layers max

print(f"Model:      {MODEL_PATH}")
print(f"Best layer: {best_layer}")
print(f"Top layers: {top_layers}")
print(f"Pairs:      {len(pairs)}")

# ── Load model (for per-category directions) ──────────────────────────────────
print(f"\nLoading model for per-category direction finding...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    output_hidden_states=True,
)
model.eval()
hidden_size = model.config.hidden_size

def get_hidden(text: str, layer: int) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=MAX_LENGTH).to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :].float().cpu()

# ── Per-category directions ───────────────────────────────────────────────────
print("\nBuilding per-category directions...")
categories_cfg = {}
categories_seen = list(set(p["category"] for p in pairs))

for category in sorted(categories_seen):
    cat_pairs = [p for p in pairs if p["category"] == category]
    if len(cat_pairs) < 5:
        print(f"  {category}: skipped (only {len(cat_pairs)} pairs)")
        continue

    h_acts, s_acts = [], []
    for p in cat_pairs:
        h_acts.append(get_hidden(p["harmful"], best_layer))
        s_acts.append(get_hidden(p["safe"],    best_layer))

    h_stack = torch.stack(h_acts)
    s_stack = torch.stack(s_acts)
    d = s_stack.mean(0) - h_stack.mean(0)
    d = d / (d.norm() + 1e-8)

    h_proj  = (h_stack @ d)
    s_proj  = (s_stack @ d)
    gap     = (s_proj.mean() - h_proj.mean()).item()
    spread  = (h_proj.std()  + s_proj.std() + 1e-8).item()
    score   = gap / spread

    categories_cfg[category] = {
        "layer":     best_layer,
        "direction": d.tolist(),
        "score":     round(score, 4),
        "n_pairs":   len(cat_pairs),
    }
    print(f"  {category:<25} score={score:.3f}  n={len(cat_pairs)}")

# ── Calibration stats ─────────────────────────────────────────────────────────
# Use global direction at best_layer to get score distribution
global_dir = directions[best_layer].float().to("cuda")
all_scores_h, all_scores_s = [], []

for p in pairs:
    h = get_hidden(p["harmful"], best_layer).to("cuda")
    s = get_hidden(p["safe"],    best_layer).to("cuda")
    all_scores_h.append((h @ global_dir).item())
    all_scores_s.append((s @ global_dir).item())

scores_h = np.array(all_scores_h)
scores_s = np.array(all_scores_s)

# Conservative default: 2 std above harmful mean
default_threshold = float(scores_h.mean() + 2.0 * scores_h.std())
hidden_norm       = float(global_dir.cpu().norm().item())

print(f"\nScore distribution:")
print(f"  Harmful  mean={scores_h.mean():.4f}  std={scores_h.std():.4f}  "
      f"p95={np.percentile(scores_h,95):.4f}")
print(f"  Safe     mean={scores_s.mean():.4f}  std={scores_s.std():.4f}  "
      f"p5={np.percentile(scores_s,5):.4f}")
print(f"  Default threshold (2σ): {default_threshold:.4f}")

# ── Ensure direction: higher score = more harmful ────────────────────────────
if scores_h.mean() < scores_s.mean():
    global_dir = -global_dir
    scores_h   = -scores_h
    scores_s   = -scores_s
    print("  Direction negated: higher score now = more harmful")

# ── Write probe config ────────────────────────────────────────────────────────
probe_cfg = {
    "model":           MODEL_PATH,
    "version":         "v1",
    "best_layer":      best_layer,
    "top_layers":      top_layers,
    "hidden_size":     hidden_size,
    "global_direction": global_dir.cpu().tolist(),
    "threshold_block": round(default_threshold, 4),
    "threshold_allow": round(default_threshold - 0.15, 4),
    "hidden_norm":     round(hidden_norm, 4),
    "score_stats": {
        "harmful_mean": round(float(scores_h.mean()), 4),
        "harmful_std":  round(float(scores_h.std()),  4),
        "safe_mean":    round(float(scores_s.mean()),  4),
        "safe_std":     round(float(scores_s.std()),   4),
    },
    "categories": categories_cfg,
    "calibrated":  False,   # step4 will set this to True
}

out_path = ARTIFACT_DIR / "probe_config.json"
with open(out_path, "w") as f:
    json.dump(probe_cfg, f, indent=2)

print(f"\nProbe config saved: {out_path}")
print(f"  Categories: {len(categories_cfg)}")
print(f"  Threshold (pre-calibration): {default_threshold:.4f}")
print(f"\nNote: run step4_calibrate.py to optimize the threshold on held-out data.")

del model; gc.collect(); torch.cuda.empty_cache()
print("\nStep 3 complete.")
# PATCH: applied after model save
