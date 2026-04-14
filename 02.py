"""
STEP 2 — Find Refusal Directions
==================================
Reads:   data/probe_pairs.json        (from step1)
Produces: artifacts/directions.pt     (tensor: layer → direction vector)
          artifacts/layer_scores.json (signal strength per layer)
          artifacts/top_layers.json   (ranked layers above threshold)

Usage:
  python step2_find_directions.py
  MODEL=votal-ai/vai35-9B python step2_find_directions.py
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
SCORE_THRESH = float(os.environ.get("SCORE_THRESH", "0.8"))
MAX_LENGTH   = 512

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    output_hidden_states=True,
)
model.eval()
n_layers    = len(model.model.layers)
hidden_size = model.config.hidden_size
print(f"  Layers: {n_layers}  Hidden: {hidden_size}")

# ── Load pairs ────────────────────────────────────────────────────────────────
with open(DATA_DIR / "probe_pairs.json") as f:
    pairs = json.load(f)
print(f"  Pairs loaded: {len(pairs)}")

# ── Extract hidden states ─────────────────────────────────────────────────────
def get_hidden_states(text: str) -> list[torch.Tensor]:
    """Return list of (hidden_size,) tensors, one per layer, at last token."""
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=MAX_LENGTH
    ).to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [h[0, -1, :].float().cpu() for h in out.hidden_states]

# ── Collect activations ───────────────────────────────────────────────────────
print("\nCollecting activations...")
harmful_acts = defaultdict(list)
safe_acts    = defaultdict(list)

for i, pair in enumerate(pairs):
    if i % 20 == 0:
        print(f"  {i}/{len(pairs)}")
    h_states = get_hidden_states(pair["harmful"])
    s_states = get_hidden_states(pair["safe"])
    for li, (h, s) in enumerate(zip(h_states, s_states)):
        harmful_acts[li].append(h)
        safe_acts[li].append(s)

# ── Compute directions ────────────────────────────────────────────────────────
print("\nComputing refusal directions per layer...")
refusal_directions = {}
layer_scores       = {}

for li in range(n_layers + 1):   # +1 for embedding layer
    if li not in harmful_acts:
        continue
    h_stack = torch.stack(harmful_acts[li])   # [N, hidden]
    s_stack = torch.stack(safe_acts[li])

    # Direction = mean(safe) - mean(harmful), normalized
    direction = s_stack.mean(0) - h_stack.mean(0)
    direction = direction / (direction.norm() + 1e-8)
    refusal_directions[li] = direction

    # Separability score: gap / spread
    h_proj = h_stack @ direction
    s_proj = s_stack @ direction
    gap    = s_proj.mean() - h_proj.mean()
    spread = h_proj.std() + s_proj.std() + 1e-8
    layer_scores[li] = (gap / spread).item()

# ── Rank layers ───────────────────────────────────────────────────────────────
ranked = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
top_layers = [li for li, sc in ranked if sc > SCORE_THRESH]
best_layer = ranked[0][0]

print(f"\n{'Layer':>6}  {'Score':>8}  {'Signal'}")
print(f"{'------':>6}  {'--------':>8}  {'──────────────────'}")
for li, sc in ranked[:15]:
    bar     = "█" * min(20, max(0, int(sc * 4)))
    marker  = " ← BEST" if li == best_layer else (" ← USE" if sc > 1.5 else "")
    print(f"{li:6d}  {sc:8.3f}  {bar}{marker}")

print(f"\nTop layers (score > {SCORE_THRESH}): {top_layers}")
print(f"Best layer: {best_layer}  score={layer_scores[best_layer]:.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(refusal_directions, ARTIFACT_DIR / "directions.pt")

with open(ARTIFACT_DIR / "layer_scores.json", "w") as f:
    json.dump({str(k): v for k, v in layer_scores.items()}, f, indent=2)

with open(ARTIFACT_DIR / "top_layers.json", "w") as f:
    json.dump({
        "top_layers":  top_layers,
        "best_layer":  best_layer,
        "best_score":  round(layer_scores[best_layer], 4),
        "score_thresh": SCORE_THRESH,
        "n_pairs":     len(pairs),
        "model":       MODEL_PATH,
    }, f, indent=2)

print(f"\nSaved to {ARTIFACT_DIR}:")
for p in ARTIFACT_DIR.iterdir():
    print(f"  {p.name}  ({p.stat().st_size/1024:.1f} KB)")

del model; gc.collect(); torch.cuda.empty_cache()
print("\nStep 2 complete.")
