#!/usr/bin/env python3
"""Reproduce Figure 4 — per-level metric heatmap.

Reads one or more prediction CSVs produced by
``scripts/stage2_infer.py`` (or any CSV with ``original_idx``,
``gt_grade``, ``final_level`` columns) and renders a 2x2 heatmap of
per-level Accuracy, Sensitivity, Precision, and F1 across the supplied
model variants. The cell holding the column-wise maximum is outlined in
red, matching the styling of the paper figure.

Numbers are computed on the fly via ``drktas.metrics.per_class_metrics``,
so the figure stays in sync with whatever predictions the script is
pointed at.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Render without a display.
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from drktas.metrics import per_class_metrics  # noqa: E402


# =============================================================================
# Computation
# =============================================================================

def per_level_accuracy(reference: np.ndarray, predicted: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """Per-class accuracy: rate of "label is correctly assigned or not".

    Following the convention used by the paper figure, this is the
    one-vs-rest accuracy for each class.
    """
    accuracy = np.zeros(num_classes, dtype=np.float64)
    for k in range(1, num_classes + 1):
        match = (reference == k) == (predicted == k)
        accuracy[k - 1] = match.mean() if match.size else 0.0
    return accuracy


def compute_panel_arrays(
    predictions: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    num_classes: int = 5,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(model_names, acc, sens, prec, f1)`` arrays of shape (M, K)."""
    names: List[str] = []
    acc_rows: List[np.ndarray] = []
    sens_rows: List[np.ndarray] = []
    prec_rows: List[np.ndarray] = []
    f1_rows: List[np.ndarray] = []

    for name, reference, predicted in predictions:
        per_class = per_class_metrics(reference, predicted, num_classes=num_classes)
        names.append(name)
        acc_rows.append(per_level_accuracy(reference, predicted, num_classes))
        sens_rows.append(per_class["sensitivity"])
        prec_rows.append(per_class["precision"])
        f1_rows.append(per_class["f1"])

    return (
        names,
        np.stack(acc_rows) * 100,
        np.stack(sens_rows) * 100,
        np.stack(prec_rows) * 100,
        np.stack(f1_rows) * 100,
    )


# =============================================================================
# I/O
# =============================================================================

def parse_predictions_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"Expected 'name=path', got {spec!r}.")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


def load_pair(path: Path, label_col: str, pred_col: str, index_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    for col in (label_col, pred_col, index_col):
        if col not in df.columns:
            raise SystemExit(f"{path}: missing column {col!r}.")
    df = df.dropna(subset=[label_col, pred_col]).sort_values(by=index_col)
    return (
        df[label_col].astype(np.int64).to_numpy(),
        df[pred_col].astype(np.int64).to_numpy(),
    )


# =============================================================================
# Rendering
# =============================================================================

def render_figure(
    names: List[str],
    panels: List[Tuple[str, np.ndarray]],
    output_base: Path,
    grades: Sequence[str] = ("L1", "L2", "L3", "L4", "L5"),
) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 14

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
    axes = axes.ravel()

    cmap = "viridis"
    vmin, vmax = 0.0, 100.0

    for ax, (title, data) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, pad=12)
        ax.set_xlabel("Level", labelpad=12)
        ax.set_xticks(np.arange(len(grades)))
        ax.set_xticklabels(grades)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names)

        ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2, alpha=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=15,
                    fontweight="bold",
                )

        # Outline the column-wise maximum
        col_max = data.max(axis=0)
        for j in range(data.shape[1]):
            for i in np.where(np.isclose(data[:, j], col_max[j]))[0]:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="red",
                        linewidth=3.0,
                    )
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=12)

    fig.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.08, wspace=0.45, hspace=0.38)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        out_path = output_base.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
        else:
            fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved {output_base.with_suffix('.png')} / .pdf / .svg")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figure 4 — per-level metric heatmap")
    parser.add_argument(
        "--predictions",
        nargs="+",
        type=parse_predictions_spec,
        required=True,
        help="One or more 'name=path/to/predictions.csv' entries.",
    )
    parser.add_argument("--index_column", type=str, default="original_idx")
    parser.add_argument("--label_column", type=str, default="gt_grade")
    parser.add_argument("--prediction_column", type=str, default="final_level")
    parser.add_argument(
        "--output",
        type=str,
        default="figures/out/fig4_per_level",
        help="Output base path (extension is added automatically).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pairs: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for name, path in args.predictions:
        ref, pred = load_pair(path, args.label_column, args.prediction_column, args.index_column)
        pairs.append((name, ref, pred))

    names, acc, sens, prec, f1 = compute_panel_arrays(pairs)
    panels = [
        ("Accuracy (%)", acc),
        ("Sensitivity (%)", sens),
        ("Precision (%)", prec),
        ("F1 score (%)", f1),
    ]

    render_figure(names, panels, Path(args.output))


if __name__ == "__main__":
    main()
