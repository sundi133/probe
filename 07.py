"""
STEP 7 — Confidence Intervals (Fixed)
Uses pre-scored data, correct threshold polarity.
"""
import json, os
import numpy as np
from pathlib import Path

ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "./artifacts"))
DATA_DIR     = Path(os.environ.get("DATA_DIR", "./data"))
N_BOOTSTRAP  = int(os.environ.get("N_BOOTSTRAP", "1000"))
CI_LEVEL     = float(os.environ.get("CI_LEVEL", "0.95"))
np.random.seed(42)

with open(ARTIFACT_DIR / "probe_config.json") as f:
    probe = json.load(f)

threshold  = probe["threshold_block"]
best_layer = probe["best_layer"]

print(f"Threshold:   {threshold}")
print(f"Bootstrap N: {N_BOOTSTRAP}")

# ── Load model and score eval set ─────────────────────────────────────────────
import torch, gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL", "votal-ai/vai35-4B")
eval_pairs = []
with open(DATA_DIR / "eval_pairs.jsonl") as f:
    for line in f:
        eval_pairs.append(json.loads(line))

print(f"Eval pairs: {len(eval_pairs)}")
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16,
    device_map="cuda", output_hidden_states=True
)
model.eval()
global_dir = torch.tensor(probe["global_direction"]).float().to("cuda")

def get_score(text):
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    return (out.hidden_states[best_layer][0,-1,:].float() @ global_dir).item()

print("Scoring...")
records = []
for i, pair in enumerate(eval_pairs):
    if i % 20 == 0: print(f"  {i}/{len(eval_pairs)}")
    records.append({
        "cat":     pair["category"],
        "score_h": get_score(pair["harmful"]),   # should be positive
        "score_s": get_score(pair["safe"]),       # should be negative
    })

del model; gc.collect(); torch.cuda.empty_cache()

# Verify polarity is correct
avg_h = np.mean([r["score_h"] for r in records])
avg_s = np.mean([r["score_s"] for r in records])
print(f"\nScore check: harmful_mean={avg_h:.3f}  safe_mean={avg_s:.3f}")
assert avg_h > avg_s, "ERROR: Direction still inverted! harmful should score higher."
print("Direction polarity: CORRECT ✓")

# Build paired arrays: each record gives one TP/TN opportunity
# y=1 for harmful, y=0 for safe
y_true_pairs   = np.array([[1, 0]] * len(records))          # shape (N, 2)
y_scores_pairs = np.array([[r["score_h"], r["score_s"]]
                             for r in records])              # shape (N, 2)

# ── Metrics on pairs ──────────────────────────────────────────────────────────
alpha = 1.0 - CI_LEVEL
lo_p  = alpha / 2 * 100
hi_p  = (1 - alpha / 2) * 100

def compute_metrics_pairs(yt, ys, t):
    """yt, ys shape (N, 2): col0=harmful, col1=safe"""
    pred_h = (ys[:, 0] >= t)   # block harmful
    pred_s = (ys[:, 1] >= t)   # should NOT block safe
    tp  = pred_h.sum()
    fn  = (~pred_h).sum()
    fp  = pred_s.sum()
    tn  = (~pred_s).sum()
    p   = tp / (tp + fp + 1e-8)
    r   = tp / (tp + fn + 1e-8)
    f1  = 2*p*r / (p + r + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    return {"precision": float(p), "recall": float(r),
            "f1": float(f1), "fpr": float(fpr),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

def auc_roc_pairs(yt, ys):
    """Flatten pairs and compute AUC."""
    y_flat = yt.flatten()
    s_flat = ys.flatten()
    idx    = np.argsort(s_flat)[::-1]
    y_s    = y_flat[idx]
    pos    = y_s.sum(); neg = len(y_s) - pos
    tp=fp=0; tpr=[0.]; fpr_=[0.]
    for lbl in y_s:
        if lbl==1: tp+=1
        else:      fp+=1
        tpr.append(tp/(pos+1e-8))
        fpr_.append(fp/(neg+1e-8))
    tpr.append(1.); fpr_.append(1.)
    return float(np.trapezoid(tpr, fpr_))

# Point estimates
pm  = compute_metrics_pairs(y_true_pairs, y_scores_pairs, threshold)
auc = auc_roc_pairs(y_true_pairs, y_scores_pairs)
print(f"\nPoint estimates (N={len(records)} pairs):")
print(f"  F1={pm['f1']:.4f}  Prec={pm['precision']:.4f}  "
      f"Rec={pm['recall']:.4f}  FPR={pm['fpr']:.4f}  AUC={auc:.4f}")

# ── Bootstrap ─────────────────────────────────────────────────────────────────
print(f"\nBootstrapping (N={N_BOOTSTRAP})...")
boot = {"precision":[],"recall":[],"f1":[],"fpr":[],"auc":[]}
n    = len(records)

for _ in range(N_BOOTSTRAP):
    idx  = np.random.choice(n, n, replace=True)
    yt_b = y_true_pairs[idx]
    ys_b = y_scores_pairs[idx]
    m    = compute_metrics_pairs(yt_b, ys_b, threshold)
    boot["precision"].append(m["precision"])
    boot["recall"].append(m["recall"])
    boot["f1"].append(m["f1"])
    boot["fpr"].append(m["fpr"])
    boot["auc"].append(auc_roc_pairs(yt_b, ys_b))

# ── Threshold stability ───────────────────────────────────────────────────────
print("Bootstrapping threshold stability...")
opt_thresholds = []
for _ in range(500):
    idx  = np.random.choice(n, n, replace=True)
    ys_b = y_scores_pairs[idx]
    yt_b = y_true_pairs[idx]
    ts   = np.linspace(ys_b.min(), ys_b.max(), 100)
    best_f1, best_t = 0.0, ts[0]
    for t in ts:
        m = compute_metrics_pairs(yt_b, ys_b, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]; best_t = t
    opt_thresholds.append(best_t)

ot = np.array(opt_thresholds)
thresh_ci = {
    "mean":   round(float(ot.mean()),4),
    "std":    round(float(ot.std()),4),
    "lo":     round(float(np.percentile(ot, lo_p)),4),
    "hi":     round(float(np.percentile(ot, hi_p)),4),
    "ci_str": f"[{np.percentile(ot,lo_p):.4f}, {np.percentile(ot,hi_p):.4f}]",
}
# Use relative stability: std relative to the score range
score_range = abs(y_scores_pairs.max() - y_scores_pairs.min())
relative_std = thresh_ci["std"] / (score_range + 1e-8)
stable = relative_std < 0.05   # stable if threshold std < 5% of score range
thresh_ci["relative_std"] = round(float(relative_std), 4)
thresh_ci["score_range"]  = round(float(score_range), 4)

# ── Per-category ──────────────────────────────────────────────────────────────
cats = sorted(set(r["cat"] for r in records))
cat_ci = {}
for cat in cats:
    idx_c = [i for i,r in enumerate(records) if r["cat"]==cat]
    if len(idx_c) < 5:
        print(f"  {cat}: skipped (n={len(idx_c)})")
        continue
    yt_c = y_true_pairs[idx_c]
    ys_c = y_scores_pairs[idx_c]
    pm_c = compute_metrics_pairs(yt_c, ys_c, threshold)
    b_f1 = []
    for _ in range(500):
        idx_b = np.random.choice(len(idx_c), len(idx_c), replace=True)
        m = compute_metrics_pairs(yt_c[idx_b], ys_c[idx_b], threshold)
        b_f1.append(m["f1"])
    b_f1 = np.array(b_f1)
    ci_str = f"[{np.percentile(b_f1,lo_p):.3f}, {np.percentile(b_f1,hi_p):.3f}]"
    cat_ci[cat] = {"f1": round(pm_c["f1"],3), "ci_str": ci_str}
    print(f"  {cat:<26} F1={pm_c['f1']:.3f}  {CI_LEVEL:.0%}CI={ci_str}")

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Overall Metrics with {CI_LEVEL:.0%} CI")
print(f"{'Metric':<12} {'Point':>8} {'Mean':>8} {'Std':>7} CI ({CI_LEVEL:.0%})")
print("─" * 55)
for metric in ["precision","recall","f1","fpr","auc"]:
    pt   = pm.get(metric, auc) if metric != "auc" else auc
    vals = np.array(boot[metric])
    ci   = f"[{np.percentile(vals,lo_p):.3f}, {np.percentile(vals,hi_p):.3f}]"
    print(f"{metric:<12} {pt:8.4f} {vals.mean():8.4f} {vals.std():7.4f} {ci}")

print(f"\nThreshold Stability:")
print(f"  Current: {threshold:.4f}")
print(f"  95% CI:  {thresh_ci['ci_str']}")
print(f"  Std:     {thresh_ci['std']:.4f}")
print(f"  Status:  {'STABLE ✓' if stable else 'UNSTABLE ✗'}")

# ── Save ──────────────────────────────────────────────────────────────────────
ci_out = {
    "threshold": threshold, "n_pairs": len(records),
    "n_bootstrap": N_BOOTSTRAP, "ci_level": CI_LEVEL,
    "overall_point": pm, "overall_auc": auc,
    "confidence_intervals": {
        k: {"mean": round(float(np.mean(v)),4),
            "std":  round(float(np.std(v)),4),
            "lo":   round(float(np.percentile(v,lo_p)),4),
            "hi":   round(float(np.percentile(v,hi_p)),4)}
        for k,v in boot.items()
    },
    "threshold_stability": thresh_ci,
    "categories": cat_ci,
}
with open(ARTIFACT_DIR / "confidence_intervals.json","w") as f:
    json.dump(ci_out, f, indent=2)
print(f"\nSaved: {ARTIFACT_DIR}/confidence_intervals.json")
