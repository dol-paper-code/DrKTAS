# Architecture

Dr.KTAS formulates KTAS classification as **hierarchical adjudication-sequence generation**. This document summarizes the components implemented in the repository.

## Problem formulation

Given a triage-time clinical record `x` (demographics, vital signs, visit info, past medical history, chief complaint, present-illness narrative), the model generates the documented KTAS adjudication tuple

```
y = (a, m, s, d, l)
```

where `a` is the age group, `m` is the chief-complaint category, `s` is the subcategory, `d` is the clinical modifier, and `l` is the final triage level (1-5). The adult and pediatric guidelines define 2,016 and 2,351 classification rules, respectively, and each entry maps a `(category, subcategory, modifier)` combination deterministically to a level.

## Stage 1 — Dual-Head model

The Stage 1 model adapts a Ministral-8B-Instruct backbone with 4-bit QLoRA and exposes two output heads from a shared representation:

- **Generation head.** The standard causal-LM head. Given the input record terminated by the delimiter `[KTAS sequence]`, it generates `y` autoregressively. Only the response tokens (after the delimiter) contribute to the generation loss; prompt tokens are masked with `-100`.
- **Classification head.** A three-layer MLP (`4096 -> 512 -> 256 -> 5`) attached to the hidden state at the delimiter position. It provides an independent ordinal prediction of the final level.

The training objective is

```
L_total = L_gen + alpha * L_cls,                       alpha = 0.5
L_cls   = L_WCE(z, y) + lambda_ord * sum_k p_k * |k - y| / 4,   lambda_ord = 0.5
```

with class weights computed by inverse-frequency reweighting (`gamma = 1.0`, unit mean normalization). Configuration knobs `target_format` and `use_classification_head` make a single trainer cover all four paper ablations: Triage-level, Triage-full context, Classification-only, and Dual-Head.

## Stage 2 — Guideline-Informed Modifier Re-adjudication

Stage 2 is applied only to **head-disagreement** cases, where the Stage 1 generation level and classification level differ. The candidate grade range is

```
C_grade = [max(1, min(l_gen, l_cls) - 1), min(5, max(l_gen, l_cls) + 1)]
```

Candidate modifiers are retrieved from the adult or pediatric lookup table (selected by the age in the clinical note) within `C_grade`. If fewer than two candidates are retrieved, the Stage 1 generation output is retained. Otherwise, a separate LoRA adapter on the same backbone generates a JSON response whose `"selection"` field picks among the candidates. Greedy decoding is used at inference; only the selection-number tokens contribute to the Stage 2 loss during training.

## Acuity-preserving gate

The Stage 2 selection is converted to a level using the deterministic modifier-to-level mapping and accepted only when it preserves or increases acuity:

```
hat{l}_DrKTAS = hat{l}_S2  if routed, valid, and hat{l}_S2 <= hat{l}_gen,
                hat{l}_gen otherwise.
```

For head-agreement cases, candidate-retrieval failures, invalid Stage 2 parses, and routed cases where the Stage 2 level is less urgent than Stage 1, the Stage 1 generation output is retained.

## Why this design

KTAS is a guideline-defined hierarchical adjudication procedure, not a single-step label. Token-level supervision over the full adjudication sequence distributes the training signal across the age group, category, subcategory, modifier, and level. This is especially relevant under severe class imbalance (Level 1 prevalence: 0.49% in the training cohort). Hierarchy-level errors can be localized to specific steps; the paper shows over half of triage-level errors first diverge at the clinical modifier step, which motivates the targeted Stage 2 design.
