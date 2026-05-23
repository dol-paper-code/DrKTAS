"""Dr.KTAS — LLM-based hierarchical KTAS triage classification.

This package contains the shared building blocks used by the Stage 1 trainer,
the Stage 2 data preparation and trainer, the Stage 2 inference pipeline, and
the baseline evaluation scripts:

* ``guidelines``  — KTAS adult/pediatric lookup access and candidate retrieval.
* ``data_io``     — CSV ingestion, prompt construction, KTAS sequence parsing.
* ``models``      — Dual-head LLM (generation head + ordinal classification
                    head) on top of a LoRA-adapted backbone.
* ``losses``      — Weighted cross-entropy with ordinal penalty, uncertainty
                    weighting, and the Stage 1 loss combiner.
* ``stage2_gate`` — Stage 2 candidate range, modifier-to-level mapping, and
                    the acuity-preserving acceptance gate.
* ``metrics``     — Agreement, ordinal metrics (QWK / kappa / MAE),
                    safety-oriented directional errors, and high-acuity
                    recall, with paired case-level bootstrap confidence
                    intervals.
* ``prompts``     — Single-source loader for the prompt templates kept
                    under ``prompts/``.

The reference for the implemented behavior is the accompanying paper
"Emergency Department Triage Classification Using LLM-Based Hierarchical
Sequence Generation" (Park et al., IEEE J-BHI, 2026).
"""

__all__ = [
    "guidelines",
    "data_io",
    "models",
    "losses",
    "stage2_gate",
    "metrics",
    "prompts",
]

__version__ = "0.1.0"
