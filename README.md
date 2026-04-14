# Adversarial Probe Pipeline

End-to-end pipeline: dataset → directions → probe → calibration → eval → CIs.

## Files

```
step1_generate_dataset.py     Generate adversarial pairs (19 categories)
step2_find_directions.py      Find refusal direction vectors per layer
step3_build_probe.py          Build probe_config.json
step4_calibrate.py            Calibrate threshold on held-out eval set
step5_eval_per_category.py    Confusion matrix per attack category
step6_regression_test.py      Ensure safe prompts are not blocked (FPR check)
step7_confidence_intervals.py Bootstrap 95% CIs for all metrics
run_pipeline.sh               Run all steps in order
```

## Quickstart

```bash
# Seed pairs only (190 pairs, no API key needed)
MODEL=votal-ai/vai35-4B ./run_pipeline.sh

# With LLM augmentation (~475 pairs)
ANTHROPIC_API_KEY=sk-ant-... MODEL=votal-ai/vai35-4B ./run_pipeline.sh

# Custom paths
MODEL=votal-ai/vai35-9B \
DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/artifacts \
./run_pipeline.sh

# Re-run specific steps only
STEPS="4 5 6 7" ./run_pipeline.sh

# Prioritise recall (e.g. for safety-critical deployment)
TARGET_RECALL=0.95 STEPS="4" ./run_pipeline.sh
```

## Output

```
artifacts/
├── probe_config.json          ← DEPLOY THIS to inference server
├── directions.pt              ← raw direction tensors
├── layer_scores.json          ← signal strength per layer
├── top_layers.json            ← ranked layers
├── calibration.json           ← threshold sweep data
├── eval_report.json           ← per-category confusion matrix
├── regression_report.json     ← false positive report
├── confidence_intervals.json  ← bootstrap CI data
└── ci_report.txt              ← human-readable summary

data/
├── adversarial_pairs.jsonl    ← all pairs
├── train_pairs.jsonl          ← 80% used for direction finding
├── eval_pairs.jsonl           ← 20% held out for calibration/testing
└── probe_pairs.json           ← flat format for step2
```

## Pass/Fail Criteria

| Metric    | Minimum | Target |
|-----------|---------|--------|
| AUC-ROC   | 0.90    | 0.95+  |
| F1        | 0.80    | 0.90+  |
| FPR       | < 10%   | < 5%   |
| Threshold CI std | < 0.10 | < 0.05 |

## Speed vs SFT

| Method         | Latency   | Speedup |
|----------------|-----------|---------|
| SFT generation | 200-400ms | 1x      |
| Probe          | 50-80ms   | 4-6x    |
| Probe + early exit | 15-30ms | 10-15x |

## Attack Categories Covered

1.  simple_instruction     — Bare direct asks
2.  context_ignoring       — Ignore all previous instructions
3.  compound_instruction   — Harmful ask buried in benign task
4.  special_case           — Claimed exceptions and clearances
5.  few_shot_attack        — Fabricated prior compliance
6.  refusal_suppression    — Do not refuse, threats
7.  context_switching      — Bait and pivot
8.  task_deflection        — Textbook/quiz/warning label cover
9.  payload_splitting      — Fragments across turns
10. defined_dictionary     — Redefined terms
11. indirect_injection     — Via URLs, emails, JSON
12. recursive_injection    — Nested prompt-in-prompt
13. code_injection         — Hidden in function stubs
14. virtualization         — Sandbox/terminal/simulation
15. pretending             — Character/entity roleplay
16. alignment_hacking      — Exploit helpfulness drives
17. authorized_user        — Fake tokens, admin claims
18. bad_chain              — False syllogisms
19. image_prompting        — Simulated OCR, fake uploads
