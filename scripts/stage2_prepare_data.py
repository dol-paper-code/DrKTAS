#!/usr/bin/env python3
"""Build the Stage 2 fine-tuning data for Dr.KTAS.

Consumes a CSV of Stage 1 (Dual-Head) predictions on the training cohort
and produces ``train.jsonl`` / ``val.jsonl`` chat records that the Stage 2
LoRA adapter is fine-tuned on. Each record is::

    {"messages": [
        {"role": "user",      "content": "<prompt with candidates>"},
        {"role": "assistant", "content": "{...,\"선택\": \"N\",\"이유\": \"...\"}"}
    ]}

Selection logic — paper, Section III-E
--------------------------------------

1. Restrict to **head-disagreement** cases (``gen_grade != cls_grade``).
2. Parse the Stage 1 generation output into ``(category, subcategory)``.
3. Retrieve modifier candidates from the adult/pediatric guideline within
   the boundary-expanded grade range from equation (5):

       C_grade = [max(1, min(gen, cls) - 1), min(5, max(gen, cls) + 1)]

   The candidate grade range is the same at training and inference, so
   that the Stage 2 adapter sees an identical candidate distribution in
   both regimes.
4. Drop cases with fewer than two retrieved candidates.
5. Drop cases whose documented (reference) modifier is not present in the
   candidate set (equivalently, the reference level is outside the
   retrieved range).
6. The training target is the candidate number of the documented modifier;
   the JSON template is rendered as the assistant response.

Per the paper, this yields 27,800 eligible cases out of 28,284 head-
disagreement cases on the CNUH training cohort, which is then split
25,020 / 2,780 for training and validation.

Differences from earlier internal scripts
-----------------------------------------

This script implements the candidate range exactly as written in the
paper, i.e. ``[min(gen, cls) - 1, max(gen, cls) + 1]``, at training time
**identically** to inference. Earlier internal versions used
``gt_grade ± 1`` at training and a different range at inference, which
produced a mismatch between the training and inference candidate
distributions. The ``gt == gen`` filter present in some earlier versions
is also not part of the paper's specification and is therefore omitted
here.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Allow running from the repository root without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from drktas.data_io import extract_age_from_clinical_note  # noqa: E402
from drktas.guidelines import GuidelineHelper  # noqa: E402
from drktas.stage2_gate import candidate_grade_range  # noqa: E402


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("drktas.stage2_prepare_data")


# =============================================================================
# Stage 1 generation parsing
# =============================================================================

def _parse_generation_string(raw: str, subcategory_vocabulary: set) -> Tuple[str, str]:
    """Return ``(subcategory, modifier)`` from a Stage 1 generation output.

    Mirrors ``scripts/stage2_infer.py``: handles both the comma-separated
    layout produced by the new Stage 1 trainer and the whitespace-separated
    layout of the legacy trainer.
    """
    if not raw or raw == "nan":
        return "", ""

    parts = raw.rsplit(" ", 1)
    text = parts[0] if len(parts) == 2 and parts[1].isdigit() else raw

    comma_parts = [p.strip() for p in text.split(",")]
    if len(comma_parts) >= 4:
        return comma_parts[2], comma_parts[3]

    tokens = text.split()
    if len(tokens) >= 3:
        remaining = " ".join(tokens[2:])
        best_match: Optional[str] = None
        for sub in subcategory_vocabulary:
            if remaining.startswith(sub) and (best_match is None or len(sub) > len(best_match)):
                best_match = sub
        if best_match:
            return best_match, remaining[len(best_match):].strip()
    return "", ""


# =============================================================================
# Reference modifier matching
# =============================================================================

def find_reference_modifier_in_candidates(
    gt_fullseverity: str, candidates: List[Dict[str, Any]], gt_grade: int
) -> Optional[int]:
    """Return the 1-indexed candidate position of the documented modifier.

    Two passes:

    1. Exact suffix match: the trailing ``"<modifier> <level>"`` of
       ``gt_fullseverity`` equals the candidate's modifier + level.
    2. Containment fallback: the candidate modifier appears as a substring
       and the candidate level matches ``gt_grade``. The longest matching
       modifier wins so that "심한 호흡곤란" is preferred over "호흡곤란"
       when both appear in the candidate list.
    """
    if not gt_fullseverity:
        return None
    gt_text = " ".join(str(gt_fullseverity).split())  # collapse whitespace
    if not gt_text or gt_text == "nan":
        return None

    for i, cand in enumerate(candidates):
        modifier = str(cand["modifier"])
        cand_level = int(cand["level"])
        if cand_level == gt_grade and gt_text.endswith(f"{modifier} {cand_level}"):
            return i + 1

    best_idx: Optional[int] = None
    best_len = 0
    for i, cand in enumerate(candidates):
        modifier = str(cand["modifier"])
        cand_level = int(cand["level"])
        if cand_level == gt_grade and modifier and modifier in gt_text:
            if len(modifier) > best_len:
                best_len = len(modifier)
                best_idx = i + 1
    return best_idx


# =============================================================================
# Prompt + response
# =============================================================================

PROMPT_TEMPLATE = """당신은 응급 환자 분류 전문가입니다. 환자 기록을 분석하여 세부 분류 후보군에서 가장 적합한 후보를 찾는 것이 당신의 목표입니다.

{clinical_note}

[세부 분류 후보군]
{candidates_text}

[검증 지침]
1. 분석: 활력징후, 주호소, 현재 병력 등 환자 기록을 주의 깊게 분석하세요.
2. 결정: 세부 분류 후보군에서 환자 기록을 가장 정확히 반영하는 분류의 번호를 출력하세요.
3. 두 후보가 동등하게 적합하다고 판단되면, 환자 안전을 위해 더 응급한 분류의 번호를 출력하세요.

[출력 형식]
{{"검증결과": "수정필요", "선택": "1", "이유": "선택의 근거를 한 문장으로 요약"}}"""


def render_candidates(candidates: List[Dict[str, Any]]) -> str:
    return "\n".join(f'  {i + 1}. "{c["modifier"]}"' for i, c in enumerate(candidates))


def build_response(gt_selection: int, gt_modifier: str) -> str:
    """Render the JSON the Stage 2 model is supervised to produce.

    The free-text reason is masked with ``-100`` by the Stage 2 collator,
    so its content does not contribute to the loss; we keep it short and
    deterministic to keep token boundaries stable.
    """
    payload = {
        "검증결과": "수정필요",
        "선택": str(gt_selection),
        "이유": f"환자의 임상 정보를 고려할 때 후보 {gt_selection}번 '{gt_modifier}'이(가) 가장 적절합니다.",
    }
    return json.dumps(payload, ensure_ascii=False)


# =============================================================================
# Stats
# =============================================================================

@dataclass
class PreparationStats:
    total: int = 0
    head_agreement: int = 0
    parse_failed: int = 0
    insufficient_candidates: int = 0
    reference_outside_range: int = 0
    eligible: int = 0
    case_b: int = 0  # gen wrong, cls right
    case_c: int = 0  # gen right, cls wrong
    case_d: int = 0  # both wrong
    selection_distribution: Counter = field(default_factory=Counter)
    grade_distribution: Counter = field(default_factory=Counter)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "head_agreement": self.head_agreement,
            "parse_failed": self.parse_failed,
            "insufficient_candidates": self.insufficient_candidates,
            "reference_outside_range": self.reference_outside_range,
            "eligible": self.eligible,
            "case_breakdown_within_disagreement": {
                "B_cls_correct": self.case_b,
                "C_gen_correct": self.case_c,
                "D_both_wrong": self.case_d,
            },
            "selection_distribution": dict(self.selection_distribution),
            "grade_distribution": dict(self.grade_distribution),
        }


# =============================================================================
# Core preparation
# =============================================================================

def prepare_training_samples(
    df: pd.DataFrame,
    helper: GuidelineHelper,
    *,
    grade_expansion: int = 1,
    min_candidates: int = 2,
) -> Tuple[List[Dict[str, Any]], PreparationStats]:
    """Build training samples from Stage 1 dual-head predictions.

    The input dataframe must contain the columns ``gen_prediction``,
    ``gen_grade``, ``cls_grade``, ``gt_grade``, ``gt_fullseverity`` and
    ``clinical_note``. The selection logic follows the paper exactly: only
    head-disagreement cases, ``[min(gen,cls)-1, max(gen,cls)+1]`` candidate
    range at training time (identical to inference), at least two retrieved
    candidates, and the documented modifier inside the retrieved set.
    """
    samples: List[Dict[str, Any]] = []
    stats = PreparationStats()

    for idx, row in df.iterrows():
        stats.total += 1

        clinical_note = str(row.get("clinical_note", "") or "")
        gen_prediction = str(row.get("gen_prediction", "") or "")
        gen_grade = int(row.get("gen_grade", 0))
        cls_grade = int(row.get("cls_grade", 0))
        gt_grade = int(row.get("gt_grade", 0))
        gt_fullseverity = str(row.get("gt_fullseverity", "") or "")

        # 1) head-disagreement only
        if gen_grade == cls_grade:
            stats.head_agreement += 1
            continue

        # 2) parse Stage 1 generation into (subcategory, modifier)
        subcategory, _ = _parse_generation_string(
            gen_prediction, helper.subcategory_vocabulary
        )
        if not subcategory:
            stats.parse_failed += 1
            continue

        age = extract_age_from_clinical_note(clinical_note)

        # 3) candidate grade range from Eq. (5) — same at train and infer
        lo, hi = candidate_grade_range(gen_grade, cls_grade, expansion=grade_expansion)

        candidates = [
            c
            for c in helper.get_modifier_candidates(subcategory, age=age)
            if lo <= int(c["level"]) <= hi
        ]

        # 4) at least two candidates
        if len(candidates) < min_candidates:
            stats.insufficient_candidates += 1
            continue

        # 5) reference modifier must lie in the candidate set
        if not (lo <= gt_grade <= hi):
            stats.reference_outside_range += 1
            continue
        gt_selection = find_reference_modifier_in_candidates(
            gt_fullseverity, candidates, gt_grade
        )
        if gt_selection is None:
            stats.reference_outside_range += 1
            continue

        gt_modifier = str(candidates[gt_selection - 1]["modifier"])

        prompt = PROMPT_TEMPLATE.format(
            clinical_note=clinical_note,
            candidates_text=render_candidates(candidates),
        )
        response = build_response(gt_selection, gt_modifier)

        # case bookkeeping (within the disagreement subset)
        if gt_grade == cls_grade:
            stats.case_b += 1
        elif gt_grade == gen_grade:
            stats.case_c += 1
        else:
            stats.case_d += 1

        stats.selection_distribution[gt_selection] += 1
        stats.grade_distribution[gt_grade] += 1
        stats.eligible += 1

        samples.append(
            {
                "original_idx": int(idx),
                "prompt": prompt,
                "response": response,
                "gt_grade": gt_grade,
                "gen_grade": gen_grade,
                "cls_grade": cls_grade,
                "subcategory": subcategory,
                "gt_modifier": gt_modifier,
                "gt_selection": gt_selection,
                "num_candidates": len(candidates),
                "age": age,
                "is_pediatric": helper.is_pediatric(age),
            }
        )

    return samples, stats


# =============================================================================
# JSONL output
# =============================================================================

def write_chat_jsonl(
    samples: List[Dict[str, Any]],
    output_path: Path,
    *,
    include_metadata: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            entry: Dict[str, Any] = {
                "messages": [
                    {"role": "user", "content": sample["prompt"]},
                    {"role": "assistant", "content": sample["response"]},
                ]
            }
            if include_metadata:
                entry["metadata"] = {
                    "gt_grade": sample["gt_grade"],
                    "gen_grade": sample["gen_grade"],
                    "cls_grade": sample["cls_grade"],
                    "subcategory": sample["subcategory"],
                    "gt_selection": sample["gt_selection"],
                    "num_candidates": sample["num_candidates"],
                    "original_idx": sample["original_idx"],
                }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Wrote %d samples to %s", len(samples), output_path)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 data preparation for Dr.KTAS")
    parser.add_argument(
        "--train_predictions",
        type=str,
        required=True,
        help="CSV of Stage 1 dual-head predictions on the training cohort.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--guidelines_dir",
        type=str,
        default=str(_REPO_ROOT / "guidelines"),
        help="Directory containing ktas_{adult,children}_guideline_lookup_clean.json.",
    )
    parser.add_argument(
        "--grade_expansion",
        type=int,
        default=1,
        help="Expansion margin used in the candidate grade range (paper: 1).",
    )
    parser.add_argument(
        "--min_candidates",
        type=int,
        default=2,
        help="Minimum number of retrieved candidates required (paper: 2).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of eligible samples placed in val.jsonl (paper: ~0.1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include_metadata",
        action="store_true",
        help="Embed bookkeeping fields under a 'metadata' key in each JSONL row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    guidelines_dir = Path(args.guidelines_dir)
    helper = GuidelineHelper(
        guidelines_dir / "ktas_adult_guideline_lookup_clean.json",
        guidelines_dir / "ktas_children_guideline_lookup_clean.json",
    )
    logger.info(
        "Loaded guideline: adult=%d children=%d subcategories",
        len(helper.adult_subcategory_to_modifiers),
        len(helper.children_subcategory_to_modifiers),
    )

    df = pd.read_csv(args.train_predictions)
    logger.info(
        "Loaded %d Stage 1 predictions from %s", len(df), args.train_predictions
    )

    samples, stats = prepare_training_samples(
        df,
        helper,
        grade_expansion=args.grade_expansion,
        min_candidates=args.min_candidates,
    )
    summary = stats.summary_dict()
    logger.info(
        "Eligible samples: %d (head_agreement_skipped=%d, parse_failed=%d, "
        "insufficient_candidates=%d, reference_outside_range=%d)",
        summary["eligible"],
        summary["head_agreement"],
        summary["parse_failed"],
        summary["insufficient_candidates"],
        summary["reference_outside_range"],
    )

    # Shuffle then split into train / val with a fixed seed so the split is
    # deterministic across runs of the script.
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_val = int(round(len(samples) * args.val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    logger.info("Split: train=%d val=%d", len(train_samples), len(val_samples))

    write_chat_jsonl(
        train_samples, output_dir / "train.jsonl", include_metadata=args.include_metadata
    )
    write_chat_jsonl(
        val_samples, output_dir / "val.jsonl", include_metadata=args.include_metadata
    )

    stats_path = output_dir / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "input": args.train_predictions,
                "grade_expansion": args.grade_expansion,
                "min_candidates": args.min_candidates,
                "val_ratio": args.val_ratio,
                "n_train": len(train_samples),
                "n_val": len(val_samples),
                "stats": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("Wrote preparation summary to %s", stats_path)


if __name__ == "__main__":
    main()
