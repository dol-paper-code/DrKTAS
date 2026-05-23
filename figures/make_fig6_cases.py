#!/usr/bin/env python3
"""Reproduce Figure 6 — Stage 2 re-adjudication case examples.

The figure shows two routed cases side by side: each panel walks through
the clinical note, the Stage 1 dual-head outputs, the Stage 2 candidate
selection, and the final decision after the acuity-preserving gate.

Because real clinical notes cannot be redistributed, this script accepts
a JSON config that specifies the two case panels. The included template
at ``figures/case_examples_template.json`` matches the structure of the
figure in the paper and can be edited (or replaced) without changing the
script.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402


# =============================================================================
# Styling
# =============================================================================

BORDER = "#2F2F2F"
INPUT_HEADER = "#6F7BF7"
STAGE1_HEADER = "#F35F7E"
STAGE2_HEADER = "#F39B3D"
RESULT_HEADER = "#9C9C9C"
SUCCESS_GREEN = "#2E9D57"
FAILURE_RED = "#D83A3A"
PANEL_BG = "#FCFCFE"


PanelLine = Any  # str | dict (with "text", "weight", "color")


def normalize_line(line: PanelLine) -> Tuple[str, str, str]:
    """Translate a JSON line entry into ``(text, weight, color_token)``."""
    if isinstance(line, str):
        return line, "normal", "black"
    if isinstance(line, dict):
        text = line["text"]
        weight = line.get("weight", "normal")
        color = line.get("color", "black")
        return text, weight, color
    raise ValueError(f"Unsupported line entry: {line!r}")


def resolve_color(token: str) -> str:
    return {
        "success_green": SUCCESS_GREEN,
        "failure_red": FAILURE_RED,
        "black": "black",
    }.get(token, token)


# =============================================================================
# Drawing
# =============================================================================

def wrap_lines(lines: List[PanelLine], width: int) -> List[Tuple[str, str, str]]:
    wrapped: List[Tuple[str, str, str]] = []
    for entry in lines:
        text, weight, color = normalize_line(entry)
        parts = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
        if not parts:
            parts = [text]
        for part in parts:
            wrapped.append((part, weight, color))
    return wrapped


def draw_rect(ax, x, y, w, h, fc, ec=BORDER, lw=1.15) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw))


def draw_box(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    header_h: float,
    header_color: str,
    title: str,
    lines: List[PanelLine],
    fontsize: float = 10.1,
    wrap_width: int = 68,
    left_pad: float = 0.012,
) -> None:
    draw_rect(ax, x, y, w, h, PANEL_BG)
    draw_rect(ax, x, y + h - header_h, w, header_h, header_color)
    ax.text(
        x + w / 2,
        y + h - header_h / 2,
        title,
        ha="center",
        va="center",
        color="white",
        fontsize=15,
        fontweight="bold",
    )

    wrapped = wrap_lines(lines, wrap_width)
    top = y + h - header_h - 0.012
    bottom = y + 0.012
    usable = top - bottom
    step = usable / max(len(wrapped), 1)
    for i, (text, weight, color_token) in enumerate(wrapped):
        ax.text(
            x + left_pad,
            top - i * step,
            text,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontweight=weight,
            color=resolve_color(color_token),
        )


def add_arrow(ax, xc: float, y0: float, y1: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (xc, y0),
            (xc, y1),
            arrowstyle="simple",
            mutation_scale=16,
            linewidth=0.8,
            color="black",
        )
    )


# =============================================================================
# Layout
# =============================================================================

PANEL_KEY_TO_HEADER = {
    "clinical_note": ("Clinical Note", INPUT_HEADER),
    "stage1": ("Stage 1 — Dual-Head Triage Model", STAGE1_HEADER),
    "stage2": ("Stage 2 — Guideline-Informed Modifier Re-adjudication", STAGE2_HEADER),
    "result": ("Result", RESULT_HEADER),
}

PANEL_ORDER = ("clinical_note", "stage1", "stage2", "result")
PANEL_Y = [0.68, 0.47, 0.245, 0.05]
PANEL_HEIGHTS = [0.245, 0.165, 0.195, 0.16]


def render_figure(config: Dict[str, Any], output_base: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(18, 13), dpi=220)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left_x, right_x = 0.045, 0.535
    panel_w = 0.42
    header_h = 0.038

    cases = config["cases"]
    if len(cases) != 2:
        raise SystemExit("The figure expects exactly two cases (left + right panel).")

    for column, case in zip((left_x, right_x), cases):
        for idx, key in enumerate(PANEL_ORDER):
            section = case.get(key, [])
            title, color = PANEL_KEY_TO_HEADER[key]
            font_size = 9.9 if key == "clinical_note" else 10.3
            wrap_width = 72 if key == "clinical_note" else 62
            draw_box(
                ax,
                x=column,
                y=PANEL_Y[idx],
                w=panel_w,
                h=PANEL_HEIGHTS[idx],
                header_h=header_h,
                header_color=color,
                title=title,
                lines=section,
                fontsize=font_size,
                wrap_width=wrap_width,
            )
        for i in range(len(PANEL_ORDER) - 1):
            add_arrow(
                ax,
                column + panel_w / 2,
                PANEL_Y[i] - 0.004,
                PANEL_Y[i + 1] + PANEL_HEIGHTS[i + 1] + 0.004,
            )

    footer = config.get(
        "footer",
        "Clinical-note details are retained; Stage 2 shows a concise subset of representative candidates.",
    )
    ax.text(0.5, 0.018, footer, ha="center", va="center", fontsize=10.5, color="#444444")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_base.with_suffix('.png')} and .pdf")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figure 6 — Stage 2 case-example panels")
    parser.add_argument(
        "--cases",
        type=str,
        default="figures/case_examples_template.json",
        help="JSON file containing the two case panels (see template).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/out/fig6_cases",
        help="Output base path (extension is added automatically).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    render_figure(config, Path(args.output))


if __name__ == "__main__":
    main()
