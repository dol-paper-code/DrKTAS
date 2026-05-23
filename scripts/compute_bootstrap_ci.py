#!/usr/bin/env python3
"""Reproduce the paper's bootstrap intervals and paired-difference tables.

This script collects per-case prediction CSVs (one per model variant) and
emits the agreement, ordinal, safety-oriented and high-acuity-recall
metrics with case-level percentile bootstrap confidence intervals, plus
paired-difference intervals on the same test records for any model pairs
that the user requests.

The inputs follow the format produced by ``scripts/stage2_infer.py``
(``predictions.csv`` with ``original_idx`` / ``gt_grade`` / ``final_level``
columns) and any of the per-model dumps in ``result/`` from the paper
experiments. Column names are overridable.

For zero-event severe-under-triage cells, the script reports the
one-sided 97.5% exact binomial upper bound, matching the convention used
in Table V of the paper.

Examples
--------

    # Internal CNUH report covering the four headline variants
    python scripts/compute_bootstrap_ci.py \
        --cohort CNUH \
        --predictions triage_level=runs/triage_level/predictions.csv \
                       triage_full=runs/triage_full_context/predictions.csv \
                       dual_head=runs/dual_head/predictions.csv \
                       drktas=runs/drktas/predictions.csv \
        --compare triage_level,triage_full \
        --compare dual_head,drktas \
        --output_dir runs/analysis_cnuh

    # External WKU report
    python scripts/compute_bootstrap_ci.py \
        --cohort WKU \
        --predictions triage_level=runs_wku/triage_level/predictions.csv \
                       triage_full=runs_wku/triage_full_context/predictions.csv \
                       dual_head=runs_wku/dual_head/predictions.csv \
                       drktas=runs_wku/drktas/predictions.csv \
        --compare triage_level,triage_full \
        --compare dual_head,drktas \
        --output_dir runs/analysis_wku
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Allow running from the repository root without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from drktas.metrics import (  # noqa: E402
    BootstrapInterval,
    any_over_triage,
    any_under_triage,
    bootstrap_metric,
    cohens_kappa,
    exact_match,
    high_acuity_recall,
    macro_metrics,
    mean_absolute_error,
    one_sided_binomial_upper_bound,
    paired_bootstrap_difference,
    per_class_metrics,
    quadratic_weighted_kappa,
    severe_under_triage,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("drktas.compute_bootstrap_ci")


# =============================================================================
# Loading / alignment
# =============================================================================

@dataclass
class ModelPredictions:
    name: str
    path: Path
    indices: np.ndarray
    reference: np.ndarray
    predicted: np.ndarray


def parse_predictions_spec(spec: str) -> Tuple[str, Path]:
    """Split a ``name=path`` CLI argument."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Expected 'name=path/to/predictions.csv', got {spec!r}."
        )
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


def load_predictions(
    name: str,
    path: Path,
    *,
    index_col: str,
    label_col: str,
    prediction_col: str,
) -> ModelPredictions:
    df = pd.read_csv(path)
    for required in (index_col, label_col, prediction_col):
        if required not in df.columns:
            raise SystemExit(
                f"Predictions file {path} is missing required column {required!r}. "
                "Use --index_column / --label_column / --prediction_column to override."
            )
    df = df.dropna(subset=[label_col, prediction_col])
    df = df.sort_values(by=index_col).reset_index(drop=True)
    return ModelPredictions(
        name=name,
        path=path,
        indices=df[index_col].astype(np.int64).to_numpy(),
        reference=df[label_col].astype(np.int64).to_numpy(),
        predicted=df[prediction_col].astype(np.int64).to_numpy(),
    )


def align_on_common_indices(
    predictions: List[ModelPredictions],
) -> Tuple[List[ModelPredictions], np.ndarray, np.ndarray]:
    """Restrict every model to the intersection of their case indices.

    Also verifies that the reference labels agree across models on each
    common case. Returns the aligned ``ModelPredictions`` list together with
    the common case indices and the shared reference array.
    """
    common = None
    for mp in predictions:
        ids = set(int(x) for x in mp.indices.tolist())
        common = ids if common is None else common & ids
    if not common:
        raise SystemExit("No common case identifiers across the supplied prediction files.")

    common_sorted = np.array(sorted(common), dtype=np.int64)
    aligned: List[ModelPredictions] = []
    reference_check: Optional[np.ndarray] = None
    for mp in predictions:
        order = {int(x): i for i, x in enumerate(mp.indices.tolist())}
        rows = np.array([order[int(idx)] for idx in common_sorted])
        ref = mp.reference[rows]
        pred = mp.predicted[rows]
        if reference_check is None:
            reference_check = ref
        elif not np.array_equal(reference_check, ref):
            mismatches = int((reference_check != ref).sum())
            raise SystemExit(
                f"Reference labels differ for {mismatches} cases between "
                f"{aligned[0].name} and {mp.name}. Ensure all CSVs use the "
                "same reference."
            )
        aligned.append(
            ModelPredictions(
                name=mp.name, path=mp.path, indices=common_sorted, reference=ref, predicted=pred
            )
        )
    assert reference_check is not None
    return aligned, common_sorted, reference_check


# =============================================================================
# Per-model report
# =============================================================================

@dataclass
class ScalarStat:
    point: float
    lower: float
    upper: float

    @classmethod
    def from_interval(cls, interval: BootstrapInterval) -> "ScalarStat":
        return cls(point=interval.point, lower=interval.lower, upper=interval.upper)

    def as_pct(self) -> Tuple[float, float, float]:
        return self.point * 100, self.lower * 100, self.upper * 100


def compute_model_report(
    mp: ModelPredictions,
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    reference = mp.reference
    predicted = mp.predicted
    n = int(reference.size)

    intervals: Dict[str, BootstrapInterval] = {
        "exact_match": bootstrap_metric(
            exact_match, reference, predicted,
            n_bootstrap=n_bootstrap, confidence=confidence, random_state=seed,
        ),
        "qwk": bootstrap_metric(
            quadratic_weighted_kappa, reference, predicted,
            n_bootstrap=n_bootstrap, confidence=confidence, random_state=seed + 1,
        ),
        "kappa": bootstrap_metric(
            cohens_kappa, reference, predicted,
            n_bootstrap=n_bootstrap, confidence=confidence, random_state=seed + 2,
        ),
        "mae": bootstrap_metric(
            mean_absolute_error, reference, predicted,
            n_bootstrap=n_bootstrap, confidence=confidence, random_state=seed + 3,
        ),
        "any_under": bootstrap_metric(
            any_under_triage, reference, predicted,
            n_bootstrap=n_bootstrap, confidence=confidence, random_state=seed + 4,
        ),
        "severe_under": bootstrap_metric(
            severe_under_triage, reference, predicted,
            n_bootstrap=n_bootstrap, confidence=confidence, random_state=seed + 5,
        ),
        "any_over": bootstrap_metric(
            any_over_triage, reference, predicted,
            n_bootstrap=n_bootstrap, confidence=confidence, random_state=seed + 6,
        ),
        "high_acuity_recall": bootstrap_metric(
            high_acuity_recall, reference, predicted,
            n_bootstrap=n_bootstrap, confidence=confidence, random_state=seed + 7,
        ),
    }

    macro = macro_metrics(reference, predicted)
    per_class = per_class_metrics(reference, predicted)
    severe_events = int(np.sum(predicted - reference >= 2))
    severe_upper_one_sided = (
        one_sided_binomial_upper_bound(severe_events, n) if severe_events == 0 else None
    )
    label_counter = Counter(int(x) for x in reference.tolist())

    return {
        "name": mp.name,
        "path": str(mp.path),
        "n": n,
        "label_distribution": {f"level_{k}": int(label_counter.get(k, 0)) for k in range(1, 6)},
        "intervals": {
            key: {"point": interval.point, "lower": interval.lower, "upper": interval.upper}
            for key, interval in intervals.items()
        },
        "macro": {key: float(val) for key, val in macro.items()},
        "per_class": {
            "sensitivity": per_class["sensitivity"].tolist(),
            "precision": per_class["precision"].tolist(),
            "f1": per_class["f1"].tolist(),
        },
        "severe_under_zero_event_upper": severe_upper_one_sided,
    }


# =============================================================================
# Paired comparisons
# =============================================================================

PAIRED_METRICS: Sequence[Tuple[str, Any]] = (
    ("agreement", exact_match),
    ("qwk", quadratic_weighted_kappa),
    ("kappa", cohens_kappa),
    ("mae", mean_absolute_error),
    ("any_under", any_under_triage),
    ("severe_under", severe_under_triage),
    ("any_over", any_over_triage),
    ("high_acuity_recall", high_acuity_recall),
)


def compute_paired_report(
    mp_a: ModelPredictions,
    mp_b: ModelPredictions,
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    """Bootstrap differences ``metric(B) - metric(A)``."""
    if not np.array_equal(mp_a.indices, mp_b.indices) or not np.array_equal(
        mp_a.reference, mp_b.reference
    ):
        raise SystemExit("Paired comparison requires aligned predictions.")

    diffs: Dict[str, Dict[str, float]] = {}
    for offset, (label, metric_fn) in enumerate(PAIRED_METRICS):
        interval = paired_bootstrap_difference(
            metric_fn,
            mp_a.reference,
            mp_a.predicted,
            mp_b.predicted,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            random_state=seed + offset,
        )
        diffs[label] = {
            "point": interval.point,
            "lower": interval.lower,
            "upper": interval.upper,
        }
    return {"a": mp_a.name, "b": mp_b.name, "differences": diffs}


# =============================================================================
# Markdown rendering
# =============================================================================

def _format_pct(point: float, lower: float, upper: float) -> str:
    return f"{point*100:.2f} [{lower*100:.2f}, {upper*100:.2f}]"


def _format_qwk(point: float, lower: float, upper: float) -> str:
    return f"{point:.4f} [{lower:.4f}, {upper:.4f}]"


def render_markdown(
    cohort: str,
    model_reports: List[Dict[str, Any]],
    pair_reports: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(f"# Bootstrap-CI report — {cohort}\n")
    lines.append(
        "All intervals are 95% case-level percentile bootstrap with 10,000 resamples "
        "(default; configurable via `--n_bootstrap`)."
    )
    lines.append("")

    if model_reports:
        first = model_reports[0]
        lines.append("## Cohort\n")
        lines.append(f"- N (aligned, common to every model): **{first['n']:,}**")
        lines.append("- Label distribution (reference):")
        for k, v in first["label_distribution"].items():
            lines.append(f"  - {k}: {v:,}")
        lines.append("")

    lines.append("## Per-model overall metrics\n")
    header = (
        "| Model | Agreement (%) | QWK | Cohen κ | MAE | Macro sens (%) | Macro prec (%) | Macro F1 (%) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines.append(header)
    for rep in model_reports:
        em = rep["intervals"]["exact_match"]
        qwk = rep["intervals"]["qwk"]
        kappa = rep["intervals"]["kappa"]
        mae = rep["intervals"]["mae"]
        m = rep["macro"]
        lines.append(
            "| {name} | {em} | {qwk} | {kappa} | {mae} | {sens:.2f} | {prec:.2f} | {f1:.2f} |".format(
                name=rep["name"],
                em=_format_pct(em["point"], em["lower"], em["upper"]),
                qwk=_format_qwk(qwk["point"], qwk["lower"], qwk["upper"]),
                kappa=f"{kappa['point']:.4f} [{kappa['lower']:.4f}, {kappa['upper']:.4f}]",
                mae=f"{mae['point']:.4f} [{mae['lower']:.4f}, {mae['upper']:.4f}]",
                sens=m["macro_sensitivity"] * 100,
                prec=m["macro_precision"] * 100,
                f1=m["macro_f1"] * 100,
            )
        )

    lines.append("")
    lines.append("## Safety-oriented metrics\n")
    header = (
        "| Model | Any under (%) | Severe under (%) | Any over (%) | High-acuity recall (%) |\n"
        "|---|---:|---:|---:|---:|"
    )
    lines.append(header)
    for rep in model_reports:
        intervals = rep["intervals"]
        au = intervals["any_under"]
        su = intervals["severe_under"]
        ao = intervals["any_over"]
        hr = intervals["high_acuity_recall"]
        if rep.get("severe_under_zero_event_upper") is not None:
            su_str = (
                f"0.00 [0.00, {rep['severe_under_zero_event_upper']*100:.2f}]"
                + "*"
            )
        else:
            su_str = _format_pct(su["point"], su["lower"], su["upper"])
        lines.append(
            "| {name} | {au} | {su} | {ao} | {hr} |".format(
                name=rep["name"],
                au=_format_pct(au["point"], au["lower"], au["upper"]),
                su=su_str,
                ao=_format_pct(ao["point"], ao["lower"], ao["upper"]),
                hr=_format_pct(hr["point"], hr["lower"], hr["upper"]),
            )
        )
    if any(rep.get("severe_under_zero_event_upper") is not None for rep in model_reports):
        lines.append("")
        lines.append("\\* Zero observed events: upper bound is one-sided 97.5% exact binomial.")
    lines.append("")

    lines.append("## Per-class (sens / prec / F1, %)\n")
    for rep in model_reports:
        lines.append(f"### {rep['name']}\n")
        lines.append("| Level | Sens (%) | Prec (%) | F1 (%) |\n|---:|---:|---:|---:|")
        sens = rep["per_class"]["sensitivity"]
        prec = rep["per_class"]["precision"]
        f1 = rep["per_class"]["f1"]
        for k in range(5):
            lines.append(
                f"| {k+1} | {sens[k]*100:.2f} | {prec[k]*100:.2f} | {f1[k]*100:.2f} |"
            )
        lines.append("")

    if pair_reports:
        lines.append("## Paired component differences\n")
        lines.append(
            "Values are `metric(B) - metric(A)` with 95% paired-bootstrap percentile intervals "
            "computed on the same test records. Percentage-point values are shown for percent metrics."
        )
        lines.append("")
        header = (
            "| Comparison (A → B) | Δ Agreement | Δ QWK | Δ Any under | Δ Severe under | Δ Over | Δ High-acuity recall |\n"
            "|---|---:|---:|---:|---:|---:|---:|"
        )
        lines.append(header)
        for pair in pair_reports:
            diffs = pair["differences"]
            ag = diffs["agreement"]
            qwk = diffs["qwk"]
            au = diffs["any_under"]
            su = diffs["severe_under"]
            ao = diffs["any_over"]
            hr = diffs["high_acuity_recall"]
            lines.append(
                "| {a} → {b} | {ag} | {qwk} | {au} | {su} | {ao} | {hr} |".format(
                    a=pair["a"],
                    b=pair["b"],
                    ag=_format_pct(ag["point"], ag["lower"], ag["upper"]),
                    qwk=_format_qwk(qwk["point"], qwk["lower"], qwk["upper"]),
                    au=_format_pct(au["point"], au["lower"], au["upper"]),
                    su=_format_pct(su["point"], su["lower"], su["upper"]),
                    ao=_format_pct(ao["point"], ao["lower"], ao["upper"]),
                    hr=_format_pct(hr["point"], hr["lower"], hr["upper"]),
                )
            )
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap-CI report for Dr.KTAS prediction CSVs."
    )
    parser.add_argument("--cohort", type=str, default="cohort")
    parser.add_argument(
        "--predictions",
        nargs="+",
        type=parse_predictions_spec,
        required=True,
        help="One or more 'name=path/to/predictions.csv' entries.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Add a paired comparison 'a,b' (metric(b) - metric(a)). May be repeated.",
    )
    parser.add_argument("--index_column", type=str, default="original_idx")
    parser.add_argument("--label_column", type=str, default="gt_grade")
    parser.add_argument("--prediction_column", type=str, default="final_level")
    parser.add_argument("--n_bootstrap", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_predictions = [
        load_predictions(
            name,
            path,
            index_col=args.index_column,
            label_col=args.label_column,
            prediction_col=args.prediction_column,
        )
        for name, path in args.predictions
    ]
    aligned, common_indices, reference = align_on_common_indices(raw_predictions)
    by_name = {mp.name: mp for mp in aligned}
    logger.info(
        "Aligned %d models on %d common cases",
        len(aligned),
        int(common_indices.size),
    )

    model_reports = [
        compute_model_report(
            mp, n_bootstrap=args.n_bootstrap, confidence=args.confidence, seed=args.seed
        )
        for mp in aligned
    ]

    pair_reports: List[Dict[str, Any]] = []
    for spec in args.compare:
        try:
            a_name, b_name = (p.strip() for p in spec.split(","))
        except ValueError as exc:
            raise SystemExit(
                f"--compare must be of the form 'a,b', got {spec!r}."
            ) from exc
        if a_name not in by_name or b_name not in by_name:
            raise SystemExit(
                f"--compare references unknown model(s): {a_name!r}, {b_name!r}. "
                f"Available: {sorted(by_name)}."
            )
        pair_reports.append(
            compute_paired_report(
                by_name[a_name],
                by_name[b_name],
                n_bootstrap=args.n_bootstrap,
                confidence=args.confidence,
                seed=args.seed + 10_000,
            )
        )

    summary = {
        "cohort": args.cohort,
        "n_common_cases": int(common_indices.size),
        "n_bootstrap": args.n_bootstrap,
        "confidence": args.confidence,
        "seed": args.seed,
        "models": model_reports,
        "paired_comparisons": pair_reports,
    }

    json_path = output_dir / f"{args.cohort.lower()}_bootstrap_analysis.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md_path = output_dir / f"{args.cohort.lower()}_bootstrap_summary.md"
    md_path.write_text(render_markdown(args.cohort, model_reports, pair_reports))

    logger.info("Wrote %s and %s", json_path, md_path)


if __name__ == "__main__":
    main()
