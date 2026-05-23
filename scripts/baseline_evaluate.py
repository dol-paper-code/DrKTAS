#!/usr/bin/env python3
"""Evaluate a zero-shot baseline prediction file.

Consumes the JSON document produced by ``scripts/baseline_infer.py`` and
reports the agreement, ordinal and safety-oriented metrics defined in
``src/drktas/metrics.py``. The same script handles single-file and sharded
(multi-rank) outputs through the ``--predictions`` argument, which accepts
a glob.

Examples
--------

    # Score a single-file inference run
    python scripts/baseline_evaluate.py --predictions runs/baseline_ministral8b.json

    # Score a torchrun multi-rank run
    python scripts/baseline_evaluate.py \
        --predictions 'runs/baseline_meditron14b_rank*.json'

    # Treat parse failures as wrong predictions (assign a fallback label)
    python scripts/baseline_evaluate.py \
        --predictions runs/baseline_ministral8b.json \
        --treat_parse_failure_as wrong --fallback_label 3
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running from the repository root without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from drktas.metrics import (  # noqa: E402
    BootstrapInterval,
    bootstrap_metric,
    exact_match,
    macro_metrics,
    one_sided_binomial_upper_bound,
    per_class_metrics,
    quadratic_weighted_kappa,
    summarize,
    severe_under_triage,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("drktas.baseline_evaluate")


# =============================================================================
# Loading
# =============================================================================

def load_predictions(pattern: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load a single file or a multi-file glob produced by baseline_infer.py.

    Returns ``(meta, results)`` where ``meta`` is the per-run configuration
    block from the first file in path-sorted order.
    """
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        # No glob characters; try as a literal path.
        candidate = Path(pattern)
        if candidate.is_file():
            paths = [candidate]
    if not paths:
        raise SystemExit(f"No prediction files match {pattern!r}.")

    meta: Optional[Dict[str, Any]] = None
    seen_indices: set = set()
    merged: List[Dict[str, Any]] = []
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if meta is None:
            meta = {k: v for k, v in doc.items() if k != "results"}
        for row in doc.get("results", []):
            idx = row.get("index")
            if idx in seen_indices:
                # Prefer the first occurrence; sharded ranks should not overlap.
                continue
            seen_indices.add(idx)
            merged.append(row)
    assert meta is not None
    logger.info(
        "Loaded %d predictions from %d file(s) for model=%s prompt_type=%s",
        len(merged),
        len(paths),
        meta.get("model"),
        meta.get("prompt_type"),
    )
    return meta, merged


# =============================================================================
# Filtering / alignment
# =============================================================================

def align_reference_and_prediction(
    rows: List[Dict[str, Any]],
    *,
    treat_parse_failure_as: str,
    fallback_label: Optional[int],
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """Build (reference, prediction) arrays and a counts dict.

    ``treat_parse_failure_as`` is one of:

    * ``'exclude'``  — drop the row (paper's primary convention).
    * ``'wrong'``    — assign ``fallback_label`` so the prediction counts as
                       a non-match.
    """
    if treat_parse_failure_as not in ("exclude", "wrong"):
        raise ValueError(
            f"treat_parse_failure_as must be 'exclude' or 'wrong', got {treat_parse_failure_as!r}."
        )

    reference: List[int] = []
    predicted: List[int] = []
    counts = {
        "total": len(rows),
        "valid": 0,
        "parse_failed": 0,
        "missing_label": 0,
        "excluded": 0,
    }

    for row in rows:
        label = row.get("true_label")
        if label is None:
            counts["missing_label"] += 1
            counts["excluded"] += 1
            continue
        pred = row.get("prediction")
        if pred is None or row.get("parse_failed"):
            counts["parse_failed"] += 1
            if treat_parse_failure_as == "exclude":
                counts["excluded"] += 1
                continue
            if fallback_label is None:
                raise SystemExit(
                    "--fallback_label is required when --treat_parse_failure_as wrong."
                )
            pred = fallback_label
        reference.append(int(label))
        predicted.append(int(pred))
        counts["valid"] += 1
    return reference, predicted, counts


# =============================================================================
# Reporting
# =============================================================================

def format_bootstrap(point: float, interval: BootstrapInterval, *, percent: bool = False) -> str:
    factor = 100.0 if percent else 1.0
    return (
        f"{point * factor:.4f} [{interval.lower * factor:.4f}, {interval.upper * factor:.4f}]"
    )


def report(
    meta: Dict[str, Any],
    reference: List[int],
    predicted: List[int],
    counts: Dict[str, int],
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    if not reference:
        raise SystemExit("No valid (reference, prediction) pairs after filtering.")

    n = len(reference)
    base = summarize(reference, predicted)

    agreement_ci = bootstrap_metric(
        exact_match, reference, predicted, n_bootstrap=n_bootstrap, confidence=confidence, random_state=seed
    )
    qwk_ci = bootstrap_metric(
        quadratic_weighted_kappa,
        reference,
        predicted,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        random_state=seed,
    )

    severe_count = int(round(base["severe_under"] * n))
    severe_ci_upper = (
        one_sided_binomial_upper_bound(severe_count, n)
        if severe_count == 0
        else None
    )

    per_class = per_class_metrics(reference, predicted)
    label_counter = Counter(reference)

    distribution = {
        f"level_{k}": int(label_counter.get(k, 0)) for k in range(1, 6)
    }

    report_doc: Dict[str, Any] = {
        "model": meta.get("model"),
        "prompt_type": meta.get("prompt_type"),
        "spec": meta.get("spec"),
        "config": meta.get("config"),
        "filter_counts": counts,
        "n_evaluated": n,
        "label_distribution": distribution,
        "agreement_pct": agreement_ci.point * 100,
        "agreement_ci_pct": [agreement_ci.lower * 100, agreement_ci.upper * 100],
        "qwk": qwk_ci.point,
        "qwk_ci": [qwk_ci.lower, qwk_ci.upper],
        "metrics": base,
        "per_class": {
            "sensitivity": per_class["sensitivity"].tolist(),
            "precision": per_class["precision"].tolist(),
            "f1": per_class["f1"].tolist(),
        },
    }
    if severe_ci_upper is not None:
        report_doc["severe_under_zero_event_ci_upper"] = severe_ci_upper
    return report_doc


def print_summary(report_doc: Dict[str, Any]) -> None:
    counts = report_doc["filter_counts"]
    base = report_doc["metrics"]
    macro = {
        k: base[k]
        for k in ("macro_sensitivity", "macro_precision", "macro_f1")
    }

    print("\n" + "=" * 72)
    print(f"Baseline:        {report_doc.get('model')} / {report_doc.get('prompt_type')}")
    print("=" * 72)
    print(
        f"Loaded:          total={counts['total']}  valid={counts['valid']}"
        f"  parse_failed={counts['parse_failed']}  excluded={counts['excluded']}"
    )
    print(f"Label dist:      {report_doc['label_distribution']}")
    print(
        f"Agreement (%):   {report_doc['agreement_pct']:.2f}"
        f"   95% CI [{report_doc['agreement_ci_pct'][0]:.2f}, {report_doc['agreement_ci_pct'][1]:.2f}]"
    )
    print(
        f"QWK:             {report_doc['qwk']:.4f}"
        f"   95% CI [{report_doc['qwk_ci'][0]:.4f}, {report_doc['qwk_ci'][1]:.4f}]"
    )
    print(f"Cohen kappa:     {base['kappa']:.4f}")
    print(f"MAE:             {base['mae']:.4f}")
    print(
        f"Macro:           sens={macro['macro_sensitivity']*100:.2f}%  "
        f"prec={macro['macro_precision']*100:.2f}%  f1={macro['macro_f1']*100:.2f}%"
    )
    print(
        f"Safety (%):      any_under={base['any_under']*100:.2f}  "
        f"severe_under={base['severe_under']*100:.2f}  any_over={base['any_over']*100:.2f}"
    )
    print(f"High-acuity rec: {base['high_acuity_recall']*100:.2f}%")

    if "severe_under_zero_event_ci_upper" in report_doc:
        upper_pct = report_doc["severe_under_zero_event_ci_upper"] * 100
        print(
            f"Severe-under is 0/{report_doc['n_evaluated']}; one-sided 97.5% upper bound = {upper_pct:.3f}%"
        )

    print("\nPer-class (sens / prec / F1, %):")
    sens = report_doc["per_class"]["sensitivity"]
    prec = report_doc["per_class"]["precision"]
    f1 = report_doc["per_class"]["f1"]
    for k in range(5):
        print(
            f"  Level {k + 1}:    {sens[k]*100:6.2f}  {prec[k]*100:6.2f}  {f1[k]*100:6.2f}"
        )
    print()


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a baseline prediction JSON.")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path or glob to baseline JSON file(s) written by baseline_infer.py.",
    )
    parser.add_argument(
        "--treat_parse_failure_as",
        choices=("exclude", "wrong"),
        default="exclude",
        help=(
            "How to handle samples where the model output could not be parsed. "
            "'exclude' drops them (paper convention); 'wrong' assigns a fallback label."
        ),
    )
    parser.add_argument(
        "--fallback_label",
        type=int,
        default=None,
        choices=(1, 2, 3, 4, 5),
        help="Fallback KTAS label used when --treat_parse_failure_as wrong.",
    )
    parser.add_argument("--n_bootstrap", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the full evaluation report as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta, rows = load_predictions(args.predictions)

    reference, predicted, counts = align_reference_and_prediction(
        rows,
        treat_parse_failure_as=args.treat_parse_failure_as,
        fallback_label=args.fallback_label,
    )

    report_doc = report(
        meta,
        reference,
        predicted,
        counts,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
        seed=args.seed,
    )
    print_summary(report_doc)

    if args.output is not None:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report_doc, indent=2, ensure_ascii=False))
        logger.info("Wrote evaluation report to %s", args.output)


if __name__ == "__main__":
    main()
