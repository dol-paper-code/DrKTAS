# Reproduction guide

This document walks through reproducing the four ablations and the final Dr.KTAS pipeline reported in the paper. All commands assume the working directory is the repository root and that `requirements.txt` has been installed.

## 0. Data and guidelines

Place your CSV cohort files outside the repository (any path) and pass them via the `--train_data` / `--test_data` / `--stage1_predictions` flags. The required CSV column schema is documented in `docs/data_governance.md`. The KTAS guideline lookup tables ship in `guidelines/`:

- `guidelines/ktas_adult_guideline_lookup_clean.json`
- `guidelines/ktas_children_guideline_lookup_clean.json`

Raw clinical notes from the paper cohorts are not redistributed; see `docs/data_governance.md`.

## 1. Stage 1 ablations (Table I)

A single Stage 1 trainer covers all four ablations through the `target_format` and `use_classification_head` knobs declared in `configs/stage1_*.yaml`. Multi-GPU launches use `torchrun`.

```bash
# Triage-level (final-digit generation)
torchrun --nproc_per_node=4 scripts/stage1_train.py \
    --config configs/stage1_triage_level.yaml \
    --train_data path/to/your_train.csv \
    --output_dir runs/stage1_triage_level

# Triage-full context (full adjudication-sequence generation)
torchrun --nproc_per_node=4 scripts/stage1_train.py \
    --config configs/stage1_triage_full_context.yaml \
    --train_data path/to/your_train.csv \
    --output_dir runs/stage1_triage_full_context

# Classification-only (ordinal classifier head; no generation loss)
torchrun --nproc_per_node=4 scripts/stage1_train.py \
    --config configs/stage1_classification_only.yaml \
    --train_data path/to/your_train.csv \
    --output_dir runs/stage1_classification_only

# Dual-Head (full-sequence generation + ordinal classification head)
torchrun --nproc_per_node=4 scripts/stage1_train.py \
    --config configs/stage1_dual_head.yaml \
    --train_data path/to/your_train.csv \
    --output_dir runs/stage1_dual_head
```

Each run writes a LoRA adapter to `<output_dir>/final_adapter/` and a periodic-checkpoint snapshot to `<output_dir>/latest_model/`.

## 2. Stage 2 — Guideline-Informed Modifier Re-adjudication (Table III)

```bash
# 1) Run Stage 1 (Dual-Head) inference on the training cohort
python scripts/stage1_infer.py \
    --config configs/stage1_dual_head.yaml \
    --lora_adapter runs/stage1_dual_head/final_adapter \
    --data path/to/your_train.csv \
    --output runs/stage1_dual_head/train_predictions.csv

# 2) Build the Stage 2 fine-tuning data from head-disagreement cases
python scripts/stage2_prepare_data.py \
    --config configs/stage2_readjudication.yaml \
    --train_predictions runs/stage1_dual_head/train_predictions.csv \
    --output_dir data_stage2

# 3) Fine-tune the Stage 2 LoRA adapter
python scripts/stage2_train.py \
    --config configs/stage2_readjudication.yaml \
    --train_data data_stage2/train.jsonl \
    --val_data data_stage2/val.jsonl \
    --output_dir runs/stage2_readjudication
```

## 3. End-to-end Dr.KTAS inference with the acuity-preserving gate

```bash
# Stage 1 (Dual-Head) inference on the internal test cohort
python scripts/stage1_infer.py \
    --config configs/stage1_dual_head.yaml \
    --lora_adapter runs/stage1_dual_head/final_adapter \
    --data path/to/your_internal_test.csv \
    --output runs/stage1_dual_head/test_predictions.csv

# Stage 2 inference + acuity-preserving gate
python scripts/stage2_infer.py \
    --config configs/stage2_readjudication.yaml \
    --stage1_predictions runs/stage1_dual_head/test_predictions.csv \
    --lora_adapter runs/stage2_readjudication/final_adapter \
    --output_dir runs/drktas_internal
```

The same recipe applied to the external cohort reproduces the same-protocol external evaluation in Tables II and V.

## 4. Evaluation

```bash
python scripts/compute_bootstrap_ci.py \
    --predictions runs/drktas_internal/predictions.csv \
    --label_column Initial_Triage_Classification \
    --output runs/drktas_internal/statistical_analysis.json \
    --n_bootstrap 10000
```

This computes the per-level metrics, ordinal metrics (QWK, Cohen's kappa, MAE), safety-oriented metrics (any/severe under-triage, any over-triage), high-acuity recall, and paired comparisons reported in the Results section.

## 5. Zero-shot baselines (Table I, rows 1-3)

```bash
# Ministral-8B zero-shot
python scripts/baseline_infer.py --model ministral8b --data path/to/your_test.csv --output runs/baseline_ministral8b.json

# Meditron3-14B zero-shot
python scripts/baseline_infer.py --model meditron14b --data path/to/your_test.csv --output runs/baseline_meditron14b.json

# GPT-4o zero-shot (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python scripts/baseline_infer.py --model gpt4o --data path/to/your_test.csv --output runs/baseline_gpt4o.json

# Score any baseline
python scripts/baseline_evaluate.py --predictions runs/baseline_gpt4o.json
```

## 6. Figures

```bash
python figures/make_fig4_per_level.py    --predictions runs/drktas_internal/predictions.csv
python figures/make_fig5_hierarchy.py    --predictions runs/drktas_internal/predictions.csv
python figures/make_fig6_cases.py        --predictions runs/drktas_internal/predictions.csv
```
