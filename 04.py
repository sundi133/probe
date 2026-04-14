"""
STEP 4 — Calibrate Threshold
==============================
Reads:   artifacts/probe_config.json
         data/eval_pairs.jsonl         (held-out from step1)
Produces: artifacts/probe_config.json  (updated with calibrated threshold)
          artifacts/calibration.json   (full calibration report)

Finds the threshold that maximises F1 on the held-out eval set.
Prints a full threshold sweep table.

Usage:
  python step4_calibrate.py
  TARGET_RECALL=0.95 python step4_calibrate.py   # prioritise recall
"""

import os, json, gc
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.environ.get("MODEL", "votal-ai/vai35-4B")
ARTIFACT_DIR  = Path(os.environ.get("ARTIFACT_DIR", "./artifacts"))
DATA_DIR      = Path(os.environ.get("DATA_DIR", "./data"))
TARGET_RECALL = float(os.environ.get("TARGET_RECALL", "0.0"))  # 0 = maximise F1
MAX_LENGTH    = 512

# ── Load ──────────────────────────────────────────────────────────────────────
with open(ARTIFACT_DIR / "probe_config.json") as f:
    probe = json.load(f)

eval_pairs = []
with open(DATA_DIR / "eval_pairs.jsonl") as f:
    for line in f:
        eval_pairs.append(json.loads(line))

print(f"Eval pairs: {len(eval_pairs)}")
print(f"Model:      {MODEL_PATH}")

# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    output_hidden_states=True,
)
model.eval()

best_layer = probe["best_layer"]
global_dir = torch.tensor(probe["global_direction"]).float().to("cuda")

def get_probe_score(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=MAX_LENGTH).to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[best_layer][0, -1, :].float()
    return (hidden @ global_dir).item()

# ── Score eval set ────────────────────────────────────────────────────────────
print("\nScoring eval set...")
y_true, y_scores = [], []

for i, pair in enumerate(eval_pairs):
    if i % 20 == 0:
        print(f"  {i}/{len(eval_pairs)}")
    score_h = get_probe_score(pair["harmful"])
    score_s = get_probe_score(pair["safe"])
    y_true.extend([1, 0])
    y_scores.extend([score_h, score_s])

y_true   = np.array(y_true)
y_scores = np.array(y_scores)

# ── AUC-ROC ───────────────────────────────────────────────────────────────────
# Manual AUC (no sklearn dependency required)
sorted_idx  = np.argsort(y_scores)[::-1]
y_sorted    = y_true[sorted_idx]
tpr_list, fpr_list = [0.0], [0.0]
pos = y_true.sum()
neg = len(y_true) - pos
tp = fp = 0
for label in y_sorted:
    if label == 1: tp += 1
    else:          fp += 1
    tpr_list.append(tp / (pos + 1e-8))
    fpr_list.append(fp / (neg + 1e-8))
tpr_list.append(1.0); fpr_list.append(1.0)
auc = np.trapz(tpr_list, fpr_list)
print(f"\nAUC-ROC: {auc:.4f}")

# ── Threshold sweep ───────────────────────────────────────────────────────────
thresholds = np.linspace(y_scores.min(), y_scores.max(), 200)
results    = []

for t in thresholds:
    preds = (y_scores >= t).astype(int)
    tp_ = int(((preds == 1) & (y_true == 1)).sum())
    fp_ = int(((preds == 1) & (y_true == 0)).sum())
    fn_ = int(((preds == 0) & (y_true == 1)).sum())
    tn_ = int(((preds == 0) & (y_true == 0)).sum())
    p_  = tp_ / (tp_ + fp_ + 1e-8)
    r_  = tp_ / (tp_ + fn_ + 1e-8)
    f1_ = 2 * p_ * r_ / (p_ + r_ + 1e-8)
    fpr_= fp_ / (fp_ + tn_ + 1e-8)
    results.append({"t": t, "tp": tp_, "fp": fp_, "fn": fn_, "tn": tn_,
                    "precision": p_, "recall": r_, "f1": f1_, "fpr": fpr_})

# ── Find optimal threshold ────────────────────────────────────────────────────
if TARGET_RECALL > 0:
    # Find threshold that achieves target recall with highest precision
    candidates = [r for r in results if r["recall"] >= TARGET_RECALL]
    if candidates:
        best = max(candidates, key=lambda x: x["precision"])
        print(f"Optimising for recall >= {TARGET_RECALL}")
    else:
        best = max(results, key=lambda x: x["f1"])
        print(f"WARNING: target recall {TARGET_RECALL} not achievable, falling back to F1")
else:
    best = max(results, key=lambda x: x["f1"])
    print(f"Optimising for F1")

# ── Print sweep table ─────────────────────────────────────────────────────────
step = max(1, len(results) // 20)
print(f"\n{'Threshold':>10} {'Prec':>7} {'Recall':>7} {'F1':>7} "
      f"{'FPR':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
print("-" * 75)
for r in results[::step]:
    marker = " ← OPTIMAL" if abs(r["t"] - best["t"]) < 1e-6 else ""
    print(f"{r['t']:10.4f} {r['precision']:7.3f} {r['recall']:7.3f} "
          f"{r['f1']:7.3f} {r['fpr']:7.3f} "
          f"{r['tp']:5d} {r['fp']:5d} {r['fn']:5d} {r['tn']:5d}{marker}")

print(f"\n{'OPTIMAL':>10} {best['precision']:7.3f} {best['recall']:7.3f} "
      f"{best['f1']:7.3f} {best['fpr']:7.3f} "
      f"{best['tp']:5d} {best['fp']:5d} {best['fn']:5d} {best['tn']:5d}")

# ── Update probe config ───────────────────────────────────────────────────────
probe["threshold_block"] = round(float(best["t"]), 4)
probe["threshold_allow"] = round(float(best["t"]) - 0.15, 4)
probe["calibrated"]      = True
probe["calibration"] = {
    "auc_roc":          round(float(auc), 4),
    "optimal_threshold":round(float(best["t"]), 4),
    "precision":        round(float(best["precision"]), 4),
    "recall":           round(float(best["recall"]), 4),
    "f1":               round(float(best["f1"]), 4),
    "fpr":              round(float(best["fpr"]), 4),
    "eval_n":           len(eval_pairs) * 2,
    "target_recall":    TARGET_RECALL,
}

with open(ARTIFACT_DIR / "probe_config.json", "w") as f:
    json.dump(probe, f, indent=2)

# Save full calibration report
cal_report = {
    "auc_roc":    round(float(auc), 4),
    "best":       best,
    "threshold_sweep": results[::step],
    "score_stats": {
        "min":  round(float(y_scores.min()), 4),
        "max":  round(float(y_scores.max()), 4),
        "mean": round(float(y_scores.mean()), 4),
        "std":  round(float(y_scores.std()), 4),
        "harmful_mean": round(float(y_scores[y_true==1].mean()), 4),
        "safe_mean":    round(float(y_scores[y_true==0].mean()), 4),
    }
}
with open(ARTIFACT_DIR / "calibration.json", "w") as f:
    json.dump(cal_report, f, indent=2)

print(f"\nUpdated probe_config.json")
print(f"  Threshold:  {probe['threshold_block']}")
print(f"  AUC-ROC:    {probe['calibration']['auc_roc']}")
print(f"  F1:         {probe['calibration']['f1']}")
print(f"  Precision:  {probe['calibration']['precision']}")
print(f"  Recall:     {probe['calibration']['recall']}")
print(f"  FPR:        {probe['calibration']['fpr']}")

del model; gc.collect(); torch.cuda.empty_cache()
print("\nStep 4 complete.")
