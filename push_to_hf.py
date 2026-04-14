#!/usr/bin/env python3
"""
push_to_hf.py — Push probe artifacts to HuggingFace
=====================================================
Usage:
  HF_TOKEN=hf_... python3 push_to_hf.py
  HF_TOKEN=hf_... REPO=votal-ai/vai35-4B-v2 python3 push_to_hf.py
  HF_TOKEN=hf_... REPO=votal-ai/vai35-9B-v2 ARTIFACT_DIR=./artifacts python3 push_to_hf.py
"""

import os, json
from pathlib import Path
from huggingface_hub import HfApi

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
REPO         = os.environ.get("REPO", "votal-ai/vai35-4B-v2")
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "./artifacts"))

if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN env var: HF_TOKEN=hf_... python3 push_to_hf.py")

# ── Load probe config for commit message ─────────────────────────────────────
probe_path = ARTIFACT_DIR / "probe_config.json"
if probe_path.exists():
    cfg = json.loads(probe_path.read_text())
    cal = cfg.get("calibration", {})
    auc = cal.get("auc_roc", "n/a")
    f1  = cal.get("f1", "n/a")
    rec = cal.get("recall", "n/a")
    fpr = cal.get("fpr", "n/a")
    msg = f"Add probe_config — AUC={auc} F1={f1} Recall={rec} FPR={fpr}"
else:
    msg = "Add probe artifacts"

print(f"Repo:    {REPO}")
print(f"Commit:  {msg}")
print()

api = HfApi()

# ── Create repo if needed ─────────────────────────────────────────────────────
api.create_repo(
    repo_id=REPO,
    token=HF_TOKEN,
    repo_type="model",
    exist_ok=True,
    private=False,
)
print(f"Repo ready: https://huggingface.co/{REPO}")

# ── Files to push ─────────────────────────────────────────────────────────────
uploads = [
    # (local_path,                          repo_path,                      required)
    (ARTIFACT_DIR / "probe_config.json",    "probe_config.json",            True),
    (ARTIFACT_DIR / "ci_report.txt",        "eval/ci_report.txt",           False),
    (ARTIFACT_DIR / "eval_report.json",     "eval/eval_report.json",        False),
    (ARTIFACT_DIR / "regression_report.json","eval/regression_report.json", False),
    (ARTIFACT_DIR / "confidence_intervals.json","eval/confidence_intervals.json", False),
    (ARTIFACT_DIR / "calibration.json",     "eval/calibration.json",        False),
    (ARTIFACT_DIR / "layer_scores.json",    "probe/layer_scores.json",      False),
    (ARTIFACT_DIR / "top_layers.json",      "probe/top_layers.json",        False),
]

pushed = []
skipped = []

for local_path, repo_path, required in uploads:
    if not local_path.exists():
        if required:
            raise FileNotFoundError(f"Required file missing: {local_path}")
        skipped.append(str(local_path.name))
        continue

    print(f"Pushing {local_path.name} → {repo_path} ...", end=" ", flush=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=REPO,
        repo_type="model",
        token=HF_TOKEN,
        commit_message=msg,
    )
    print("✓")
    pushed.append(repo_path)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print(f"Pushed {len(pushed)} files to {REPO}")
for p in pushed:
    print(f"  ✓  {p}")
if skipped:
    print(f"\nSkipped (not found): {', '.join(skipped)}")
print(f"\nRepo: https://huggingface.co/{REPO}")
print("=" * 55)

