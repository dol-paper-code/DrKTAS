"""KTAS evaluation metrics with paired bootstrap confidence intervals.

All metrics in this module are computed relative to the documented
institutional KTAS assignments, which serve as the reference target for
protocol-aligned evaluation in the paper (Section III-F).

The functions cover:

* Agreement (exact match).
* Macro-averaged per-level sensitivity, precision and F1.
* Ordinal metrics: quadratic weighted kappa, Cohen's kappa, mean absolute
  error.
* Safety-oriented directional errors: any under-triage
  (``pred > ref``), severe under-triage (``pred - ref >= 2``), any
  over-triage (``pred < ref``).
* High-acuity recall: among documented KTAS Level 1-2 encounters, the
  proportion predicted as Level 1 or Level 2.
* Case-level non-parametric bootstrap with 95% percentile intervals for any
  scalar metric, including paired-difference intervals on identical test
  records.

For zero-event rate cells (for example, severe under-triage = 0/N), the
two-sided percentile bound collapses to the trivial interval ``[0, 0]``.
The paper convention is to report the one-sided 97.5% exact binomial upper
bound in that situation. :func:`one_sided_binomial_upper_bound` implements
that fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta


# ---------------------------------------------------------------- Confusion

def _to_array(values: Iterable[int]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int64)


def _confusion_matrix(
    reference: np.ndarray,
    predicted: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Return a ``num_classes x num_classes`` confusion matrix.

    Rows are reference levels, columns are predicted levels. Labels are
    expected in ``[1, num_classes]`` (KTAS uses 1-5).
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for r, p in zip(reference, predicted):
        if 1 <= r <= num_classes and 1 <= p <= num_classes:
            cm[int(r) - 1, int(p) - 1] += 1
    return cm


# ---------------------------------------------------------------- Per-class

def per_class_metrics(
    reference: Sequence[int],
    predicted: Sequence[int],
    num_classes: int = 5,
) -> Dict[str, np.ndarray]:
    """Per-class sensitivity, precision and F1.

    Returns dictionaries with one value per class. Macro means are returned
    separately by :func:`macro_metrics`.
    """
    ref = _to_array(reference)
    pred = _to_array(predicted)
    cm = _confusion_matrix(ref, pred, num_classes)

    tp = np.diag(cm).astype(np.float64)
    row_sum = cm.sum(axis=1).astype(np.float64)  # reference per class
    col_sum = cm.sum(axis=0).astype(np.float64)  # predicted per class

    sensitivity = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum > 0)
    precision = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum > 0)
    denom = sensitivity + precision
    f1 = np.divide(2 * sensitivity * precision, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"sensitivity": sensitivity, "precision": precision, "f1": f1}


def macro_metrics(
    reference: Sequence[int],
    predicted: Sequence[int],
    num_classes: int = 5,
) -> Dict[str, float]:
    """Macro-averaged sensitivity, precision and F1."""
    per_class = per_class_metrics(reference, predicted, num_classes)
    return {
        "macro_sensitivity": float(per_class["sensitivity"].mean()),
        "macro_precision": float(per_class["precision"].mean()),
        "macro_f1": float(per_class["f1"].mean()),
    }


# ---------------------------------------------------------------- Agreement

def exact_match(reference: Sequence[int], predicted: Sequence[int]) -> float:
    ref = _to_array(reference)
    pred = _to_array(predicted)
    if ref.size == 0:
        return 0.0
    return float((ref == pred).mean())


def mean_absolute_error(reference: Sequence[int], predicted: Sequence[int]) -> float:
    ref = _to_array(reference).astype(np.float64)
    pred = _to_array(predicted).astype(np.float64)
    if ref.size == 0:
        return 0.0
    return float(np.abs(ref - pred).mean())


# ---------------------------------------------------------------- Kappa

def _kappa(
    reference: np.ndarray,
    predicted: np.ndarray,
    num_classes: int,
    weights: np.ndarray,
) -> float:
    cm = _confusion_matrix(reference, predicted, num_classes).astype(np.float64)
    n = cm.sum()
    if n == 0:
        return 0.0

    row_marginals = cm.sum(axis=1, keepdims=True) / n
    col_marginals = cm.sum(axis=0, keepdims=True) / n
    observed = cm / n
    expected = row_marginals @ col_marginals

    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return 0.0
    return 1.0 - numerator / denominator


def cohens_kappa(
    reference: Sequence[int],
    predicted: Sequence[int],
    num_classes: int = 5,
) -> float:
    ref = _to_array(reference)
    pred = _to_array(predicted)
    weights = 1.0 - np.eye(num_classes)
    return _kappa(ref, pred, num_classes, weights)


def quadratic_weighted_kappa(
    reference: Sequence[int],
    predicted: Sequence[int],
    num_classes: int = 5,
) -> float:
    ref = _to_array(reference)
    pred = _to_array(predicted)
    indices = np.arange(num_classes)
    weights = ((indices[:, None] - indices[None, :]) ** 2) / ((num_classes - 1) ** 2)
    return _kappa(ref, pred, num_classes, weights)


# ----------------------------------------------------- Safety-oriented metrics

def any_under_triage(reference: Sequence[int], predicted: Sequence[int]) -> float:
    """Fraction of cases where ``predicted > reference`` (less urgent)."""
    ref = _to_array(reference)
    pred = _to_array(predicted)
    if ref.size == 0:
        return 0.0
    return float((pred > ref).mean())


def severe_under_triage(reference: Sequence[int], predicted: Sequence[int]) -> float:
    """Fraction of cases where ``predicted - reference >= 2``."""
    ref = _to_array(reference)
    pred = _to_array(predicted)
    if ref.size == 0:
        return 0.0
    return float((pred - ref >= 2).mean())


def any_over_triage(reference: Sequence[int], predicted: Sequence[int]) -> float:
    """Fraction of cases where ``predicted < reference`` (more urgent)."""
    ref = _to_array(reference)
    pred = _to_array(predicted)
    if ref.size == 0:
        return 0.0
    return float((pred < ref).mean())


def high_acuity_recall(
    reference: Sequence[int],
    predicted: Sequence[int],
    high_acuity_levels: Sequence[int] = (1, 2),
) -> float:
    """Recall over documented KTAS Level 1-2 encounters.

    Returns the proportion of high-acuity reference cases that were
    predicted at one of the same high-acuity levels.
    """
    ref = _to_array(reference)
    pred = _to_array(predicted)
    high_set = set(int(x) for x in high_acuity_levels)
    mask = np.array([int(x) in high_set for x in ref], dtype=bool)
    if not mask.any():
        return 0.0
    matched = np.array([int(x) in high_set for x in pred])
    return float(matched[mask].mean())


# --------------------------------------------------------- Bootstrap

@dataclass(frozen=True)
class BootstrapInterval:
    """A scalar metric estimate with a non-parametric 95% interval."""

    point: float
    lower: float
    upper: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def one_sided_binomial_upper_bound(
    events: int,
    n: int,
    confidence: float = 0.975,
) -> float:
    """One-sided exact upper bound on a proportion (Clopper-Pearson).

    Used in the paper for zero-event safety cells, for which percentile
    bootstrap collapses to ``[0, 0]``. With ``events = 0`` and the default
    confidence ``0.975``, this matches the upper bound reported as
    ``0.00 [0.00, 0.03]`` in Table V.
    """
    if n <= 0:
        return 0.0
    if events >= n:
        return 1.0
    if events == 0:
        return float(1.0 - (1.0 - confidence) ** (1.0 / n))
    return float(beta.ppf(confidence, events + 1, n - events))


def bootstrap_metric(
    metric_fn: Callable[..., float],
    reference: Sequence[int],
    predicted: Sequence[int],
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
    **kwargs,
) -> BootstrapInterval:
    """Case-level percentile bootstrap for a scalar metric.

    The function resamples row indices with replacement ``n_bootstrap``
    times, recomputes the metric on each resample, and returns the lower
    and upper percentiles together with the point estimate from the
    original data. Extra keyword arguments are forwarded to ``metric_fn``.
    """
    ref = _to_array(reference)
    pred = _to_array(predicted)
    if ref.size == 0:
        return BootstrapInterval(0.0, 0.0, 0.0)

    point = float(metric_fn(ref, pred, **kwargs))
    rng = np.random.default_rng(random_state)
    n = ref.size
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(ref[idx], pred[idx], **kwargs)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return BootstrapInterval(point=point, lower=lo, upper=hi)


def paired_bootstrap_difference(
    metric_fn: Callable[..., float],
    reference: Sequence[int],
    predicted_a: Sequence[int],
    predicted_b: Sequence[int],
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
    **kwargs,
) -> BootstrapInterval:
    """Paired bootstrap for the difference ``metric(B) - metric(A)``.

    Identical row indices are resampled across both prediction vectors,
    yielding the paired-comparison intervals reported in Table III of the
    paper.
    """
    ref = _to_array(reference)
    a = _to_array(predicted_a)
    b = _to_array(predicted_b)
    if not (ref.size == a.size == b.size):
        raise ValueError("Reference and both prediction vectors must be aligned.")
    if ref.size == 0:
        return BootstrapInterval(0.0, 0.0, 0.0)

    point = float(metric_fn(ref, b, **kwargs)) - float(metric_fn(ref, a, **kwargs))
    rng = np.random.default_rng(random_state)
    n = ref.size
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(ref[idx], b[idx], **kwargs) - metric_fn(
            ref[idx], a[idx], **kwargs
        )
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return BootstrapInterval(point=point, lower=lo, upper=hi)


# ----------------------------------------------------- Composite report

def summarize(
    reference: Sequence[int],
    predicted: Sequence[int],
    num_classes: int = 5,
    high_acuity_levels: Sequence[int] = (1, 2),
) -> Dict[str, float]:
    """All scalar metrics in one call, without bootstrap intervals.

    Useful for quick comparisons. For confidence intervals see
    :func:`bootstrap_metric` and :func:`paired_bootstrap_difference`.
    """
    macro = macro_metrics(reference, predicted, num_classes)
    return {
        "agreement": exact_match(reference, predicted),
        "qwk": quadratic_weighted_kappa(reference, predicted, num_classes),
        "kappa": cohens_kappa(reference, predicted, num_classes),
        "mae": mean_absolute_error(reference, predicted),
        "any_under": any_under_triage(reference, predicted),
        "severe_under": severe_under_triage(reference, predicted),
        "any_over": any_over_triage(reference, predicted),
        "high_acuity_recall": high_acuity_recall(reference, predicted, high_acuity_levels),
        **macro,
    }


__all__: List[str] = [
    "BootstrapInterval",
    "exact_match",
    "mean_absolute_error",
    "per_class_metrics",
    "macro_metrics",
    "cohens_kappa",
    "quadratic_weighted_kappa",
    "any_under_triage",
    "severe_under_triage",
    "any_over_triage",
    "high_acuity_recall",
    "summarize",
    "bootstrap_metric",
    "paired_bootstrap_difference",
    "one_sided_binomial_upper_bound",
]
