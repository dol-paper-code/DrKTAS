#!/usr/bin/env python3
"""Reproduce Figure 5 — hierarchy-level error analysis.

Takes a Stage 1 dual-head prediction CSV that contains the generated KTAS
sequence and the documented reference sequence, decomposes the
adjudication path into ``(a, m, s, d, l)``, and emits two panels:

* **5(a) Hierarchy-level error rate** — per-position error rate of the
  Stage 1 generation head.
* **5(b) First-error origin** — among triage-level errors, the share of
  cases that first diverge at each position of the hierarchy.

Both panels are computed on the fly so the figure tracks any new
prediction CSV.

Expected CSV columns
--------------------

``gen_prediction``     — comma-separated tuple ``a, m, s, d, l`` or the
                         space-separated legacy layout.
``gt_fullseverity``    — same layout as the reference target.
``gt_grade``           — reference KTAS level (1-5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from drktas.data_io import split_fullseverity_tokens  # noqa: E402
from drktas.guidelines import GuidelineHelper  # noqa: E402


# =============================================================================
# Sequence parsing
# =============================================================================

LEVEL_LABELS = (
    "Age group (a)",
    "Major category (m)",
    "Subcategory (s)",
    "Clinical modifier (d)",
    "Severity level (ℓ)",
)


def parse_sequence(
    raw: str, subcategory_vocabulary: Optional[set] = None
) -> Optional[Tuple[str, str, str, str, int]]:
    """Best-effort parser for both comma- and whitespace-separated layouts.

    First tries the structured comma-separated form
    (``a, m, s, d, l``). Falls back to the legacy whitespace layout when
    the comma form is unavailable, using ``subcategory_vocabulary`` to
    recover the subcategory boundary.
    """
    if not raw or raw == "nan":
        return None

    parsed = split_fullseverity_tokens(raw)
    if parsed is not None:
        return parsed

    # Whitespace fallback: <age>, <category>, <subcategory>, <modifier>, <level>
    parts = raw.rsplit(" ", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    try:
        level = int(parts[1])
    except ValueError:
        return None
    if not (1 <= level <= 5):
        return None
    head_tokens = parts[0].split()
    if len(head_tokens) < 3:
        return None
    age_group, category = head_tokens[0], head_tokens[1]
    remaining = " ".join(head_tokens[2:])

    if subcategory_vocabulary:
        best_match: Optional[str] = None
        for sub in subcategory_vocabulary:
            if remaining.startswith(sub) and (best_match is None or len(sub) > len(best_match)):
                best_match = sub
        if best_match:
            modifier = remaining[len(best_match):].strip()
            return age_group, category, best_match, modifier, level
    return None


# =============================================================================
# Counting
# =============================================================================

def compute_hierarchy_breakdown(
    df: pd.DataFrame,
    subcategory_vocabulary: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return ``(accuracy_per_level, first_error_count, first_error_share, n_total, n_errors)``.

    ``accuracy_per_level`` is the share of cases where each successive
    field matches the reference, computed only over cases where the
    generation sequence was parseable.

    ``first_error_count`` counts the number of error cases whose first
    divergence happens at each level (counts sum to the number of
    triage-level errors). ``first_error_share`` is the same as fractions.
    """
    n_levels = len(LEVEL_LABELS)
    correct_per_level = np.zeros(n_levels, dtype=np.int64)
    seen_per_level = np.zeros(n_levels, dtype=np.int64)
    first_error_count = np.zeros(n_levels, dtype=np.int64)
    n_total = 0
    n_errors = 0

    for _, row in df.iterrows():
        gen = parse_sequence(str(row.get("gen_prediction", "")), subcategory_vocabulary)
        ref = parse_sequence(str(row.get("gt_fullseverity", "")), subcategory_vocabulary)
        if gen is None or ref is None:
            continue
        n_total += 1

        any_wrong = False
        first_wrong = None
        for i in range(n_levels):
            seen_per_level[i] += 1
            if gen[i] == ref[i]:
                correct_per_level[i] += 1
            else:
                if first_wrong is None:
                    first_wrong = i
                any_wrong = True

        if any_wrong:
            n_errors += 1
            first_error_count[first_wrong] += 1

    accuracy = np.divide(
        correct_per_level,
        seen_per_level,
        out=np.zeros_like(correct_per_level, dtype=np.float64),
        where=seen_per_level > 0,
    )
    if n_errors > 0:
        share = first_error_count.astype(np.float64) / n_errors
    else:
        share = np.zeros_like(first_error_count, dtype=np.float64)
    return accuracy, first_error_count, share, n_total, n_errors


# =============================================================================
# Rendering
# =============================================================================

def render_panels(
    accuracy: np.ndarray,
    first_error_count: np.ndarray,
    first_error_share: np.ndarray,
    n_errors: int,
    output_base: Path,
    title: str,
) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"

    # --- Panel (a): hierarchy-level error rate ---------------------------
    fig_a, ax = plt.subplots(figsize=(8.0, 4.8))
    error_rate = (1.0 - accuracy) * 100
    y = np.arange(len(LEVEL_LABELS))
    bars = ax.barh(y, error_rate)
    ax.set_yticks(y)
    ax.set_yticklabels(LEVEL_LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, max(60.0, float(error_rate.max()) * 1.15))
    ax.set_xlabel("Error rate (%)")
    ax.set_title(f"{title}: Hierarchy-level error rate", pad=10)
    ax.grid(axis="x", alpha=0.3)
    for bar, value in zip(bars, error_rate):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            fontsize=10,
        )
    fig_a.tight_layout()
    save_all_formats(fig_a, output_base.with_name(output_base.name + "_a"))
    plt.close(fig_a)

    # --- Panel (b): first-error origin -----------------------------------
    fig_b, ax = plt.subplots(figsize=(9.0, 5.0))
    stages = (
        "a wrong",
        "m first wrong\n(a ok)",
        "s first wrong\n(a,m ok)",
        "d first wrong\n(a,m,s ok)",
        "ℓ first wrong\n(a,m,s,d ok)",
    )
    share_pct = first_error_share * 100
    x = np.arange(len(stages))
    ax.plot(x, share_pct, marker="o", linewidth=1.8, markersize=8)
    for xi, p, c in zip(x, share_pct, first_error_count):
        y_text = p + 2.0 if p > 0 else 2.0
        ax.text(
            xi,
            y_text,
            f"{p:.1f}%\n(n={int(c):,})",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Share among misclassified cases (%)")
    ax.set_xlabel("First hierarchy level at which prediction becomes incorrect")
    upper = max(65.0, float(share_pct.max()) * 1.2 if share_pct.size else 10.0)
    ax.set_ylim(0, upper)
    ax.set_title(f"{title}: First-error origin (n_errors={n_errors:,})", pad=10)
    ax.grid(axis="y", alpha=0.3)
    fig_b.tight_layout()
    save_all_formats(fig_b, output_base.with_name(output_base.name + "_b"))
    plt.close(fig_b)


def save_all_formats(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved {base.with_suffix('.png')} / .pdf / .svg")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figure 5 — hierarchy-level error analysis")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Stage 1 dual-head prediction CSV (must contain gen_prediction + gt_fullseverity).",
    )
    parser.add_argument(
        "--guidelines_dir",
        type=str,
        default=str(_REPO_ROOT / "guidelines"),
        help="Directory with the KTAS guideline JSON files (used as a parsing fallback).",
    )
    parser.add_argument("--title", type=str, default="Dual-Head (gen. output)")
    parser.add_argument(
        "--output",
        type=str,
        default="figures/out/fig5_hierarchy",
        help="Output base path; the _a and _b suffixes are added automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions)
    for col in ("gen_prediction", "gt_fullseverity"):
        if col not in df.columns:
            raise SystemExit(f"{args.predictions}: missing column {col!r}.")

    subcategory_vocabulary: Optional[set] = None
    guidelines_dir = Path(args.guidelines_dir)
    if guidelines_dir.is_dir():
        helper = GuidelineHelper(
            guidelines_dir / "ktas_adult_guideline_lookup_clean.json",
            guidelines_dir / "ktas_children_guideline_lookup_clean.json",
        )
        subcategory_vocabulary = helper.subcategory_vocabulary

    accuracy, first_error_count, first_error_share, n_total, n_errors = compute_hierarchy_breakdown(
        df, subcategory_vocabulary
    )
    print(
        f"Parsed {n_total:,} of {len(df):,} cases; {n_errors:,} had at least one level wrong."
    )
    print("Per-level accuracy (%):", {label: round(a * 100, 2) for label, a in zip(LEVEL_LABELS, accuracy)})

    render_panels(
        accuracy,
        first_error_count,
        first_error_share,
        n_errors,
        Path(args.output),
        title=args.title,
    )


if __name__ == "__main__":
    main()
