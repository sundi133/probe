#!/bin/bash
# run_pipeline.sh — Run all 7 steps end to end
# Usage:
#   ./run_pipeline.sh
#   MODEL=votal-ai/vai35-9B ./run_pipeline.sh
#   ANTHROPIC_API_KEY=sk-ant-... ./run_pipeline.sh   # enables LLM augmentation
#   STEPS="1 2 3" ./run_pipeline.sh                  # run specific steps only
#   TARGET_RECALL=0.95 ./run_pipeline.sh             # prioritise recall in step4

set -e

# ── Config ────────────────────────────────────────────────────────────────────
export MODEL=${MODEL:-"votal-ai/vai35-4B"}
export DATA_DIR=${DATA_DIR:-"./data"}
export ARTIFACT_DIR=${ARTIFACT_DIR:-"./artifacts"}
export OUT_DIR=$DATA_DIR
export STEPS=${STEPS:-"1 2 3 4 5 6 7"}
export TARGET_RECALL=${TARGET_RECALL:-"0.95"}
export F1_WARN=${F1_WARN:-"0.85"}
export MAX_FPR=${MAX_FPR:-"0.05"}
export N_BOOTSTRAP=${N_BOOTSTRAP:-"1000"}
export CI_LEVEL=${CI_LEVEL:-"0.95"}

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[STEP $1]${NC} $2"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

PYTHON=${PYTHON:-"python3"}

mkdir -p $DATA_DIR $ARTIFACT_DIR

echo "========================================"
echo "  Adversarial Probe Pipeline"
echo "========================================"
echo "  MODEL:         $MODEL"
echo "  DATA_DIR:      $DATA_DIR"
echo "  ARTIFACT_DIR:  $ARTIFACT_DIR"
echo "  STEPS:         $STEPS"
echo "  TARGET_RECALL: $TARGET_RECALL"
echo "  MAX_FPR:       $MAX_FPR"
echo "  N_BOOTSTRAP:   $N_BOOTSTRAP"
echo "  LLM augment:   ${ANTHROPIC_API_KEY:+YES (ANTHROPIC_API_KEY set)}"
echo "========================================"
echo ""

run_step() {
    local n=$1
    local script=$2
    local desc=$3
    if echo "$STEPS" | grep -qw "$n"; then
        info $n "$desc"
        if [ ! -f "$script" ]; then
            fail "Script not found: $script"
        fi
        $PYTHON $script || fail "Step $n ($script) failed"
        echo ""
    else
        echo -e "${YELLOW}[SKIP $n]${NC} $desc"
    fi
}

# ── Run steps ─────────────────────────────────────────────────────────────────
run_step 1 step1_generate_dataset.py     "Generate adversarial dataset"
run_step 2 step2_find_directions.py      "Find refusal directions"
run_step 3 step3_build_probe.py          "Build probe config"
run_step 4 step4_calibrate.py            "Calibrate threshold (recall>=$TARGET_RECALL)"
run_step 5 step5_eval_per_category.py    "Evaluate per category (F1 warn<$F1_WARN)"
run_step 6 step6_regression_test.py      "Regression test (safe FPR<$MAX_FPR)"
run_step 7 07_fixed.py                   "Confidence intervals (N=$N_BOOTSTRAP)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo "========================================"
echo "  Pipeline complete"
echo "========================================"
echo ""
echo "Key output files:"
echo "  $ARTIFACT_DIR/probe_config.json       <- deploy this"
echo "  $ARTIFACT_DIR/ci_report.txt           <- accuracy summary"
echo "  $ARTIFACT_DIR/eval_report.json        <- per-category breakdown"
echo "  $ARTIFACT_DIR/regression_report.json  <- false positive check"
echo "  $ARTIFACT_DIR/confidence_intervals.json <- bootstrap CIs"
echo ""

# ── Print final summary ───────────────────────────────────────────────────────
ARTIFACT_DIR_PY="$ARTIFACT_DIR"
$PYTHON - << EOF
import json, sys
from pathlib import Path

art = Path("$ARTIFACT_DIR_PY")

# Probe config
probe_path = art / "probe_config.json"
if not probe_path.exists():
    print("No probe_config.json found")
    sys.exit(0)

cfg = json.loads(probe_path.read_text())
print("=" * 50)
print("Probe Config")
print("=" * 50)
print(f"  model:      {cfg.get('model', 'n/a')}")
print(f"  threshold:  {cfg.get('threshold_block', 'n/a')}")
print(f"  best_layer: {cfg.get('best_layer', 'n/a')}")
print(f"  calibrated: {cfg.get('calibrated', False)}")
print(f"  categories: {len(cfg.get('categories', {}))}")

if "calibration" in cfg:
    c = cfg["calibration"]
    print(f"\nCalibration Metrics")
    print(f"  AUC-ROC:    {c.get('auc_roc', 'n/a')}")
    print(f"  F1:         {c.get('f1', 'n/a')}")
    print(f"  Precision:  {c.get('precision', 'n/a')}")
    print(f"  Recall:     {c.get('recall', 'n/a')}")
    print(f"  FPR:        {c.get('fpr', 'n/a')}")

# Regression test
reg_path = art / "regression_report.json"
if reg_path.exists():
    rr = json.loads(reg_path.read_text())
    status = "PASS ✓" if rr["passed"] else "FAIL ✗"
    print(f"\nRegression Test: {status}")
    print(f"  Safe FPR: {rr['fpr']:.1%}  (max: {rr['max_fpr']:.1%})")

# Eval report
eval_path = art / "eval_report.json"
if eval_path.exists():
    er = json.loads(eval_path.read_text())
    ov = er.get("overall", {})
    print(f"\nPer-Category Eval")
    print(f"  Overall F1:  {ov.get('f1', 'n/a')}")
    print(f"  Weak cats:   {er.get('weak_categories', []) or 'none'}")

# CI report
ci_txt = art / "ci_report.txt"
if ci_txt.exists():
    print(f"\n{ci_txt.read_text()}")
else:
    # Try confidence_intervals.json
    ci_path = art / "confidence_intervals.json"
    if ci_path.exists():
        ci = json.loads(ci_path.read_text())
        ov = ci.get("overall_point", {})
        ts = ci.get("threshold_stability", {})
        print(f"\nConfidence Intervals (95%)")
        cis = ci.get("confidence_intervals", {})
        for m in ["f1", "precision", "recall", "fpr", "auc"]:
            if m in cis:
                d = cis[m]
                print(f"  {m:<12} {ov.get(m, ci.get('overall_auc', 'n/a')):.4f}  "
                      f"[{d['lo']:.3f}, {d['hi']:.3f}]")
        print(f"\nThreshold Stability")
        print(f"  CI:     {ts.get('ci_str', 'n/a')}")
        print(f"  Std:    {ts.get('std', 'n/a')}")
        print(f"  Status: {'STABLE ✓' if ts.get('relative_std', 1) < 0.05 else 'check ci_str'}")

print("\n" + "=" * 50)
print("Done. Deploy: artifacts/probe_config.json")
print("=" * 50)
EOF
