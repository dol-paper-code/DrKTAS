#!/usr/bin/env python3
"""Stage 2 inference for Dr.KTAS.

Given a Stage 1 (Dual-Head) prediction CSV, this script:

1. Filters head-disagreement cases (``gen_grade != cls_grade``).
2. Parses each case's generation output into ``(category, subcategory)``.
3. Retrieves modifier candidates from the adult/pediatric guideline within
   the boundary-expanded grade range
   ``[max(1, min(gen, cls) - expansion), min(5, max(gen, cls) + expansion)]``.
4. Asks the Stage 2 LoRA adapter to pick the most appropriate candidate.
5. Maps the selected modifier back to a KTAS level and applies the
   **acuity-preserving gate** from :mod:`drktas.stage2_gate`: the Stage 2
   level is accepted only when it is the same as or more urgent than the
   Stage 1 generation level. Otherwise the Stage 1 generation level is
   retained.

Outputs
-------

``predictions.csv``
    Per-case final decisions. One row per case with the columns the
    bootstrap CI script consumes (``gt_grade``, ``gen_grade``, ``cls_grade``,
    ``stage2_grade``, ``final_level``, ``source``).

``routing_summary.json``
    Aggregate routing statistics (head-disagreement count, gate accept /
    reject counts, candidate-retrieval failures, JSON-parse failures).

``verifier_logs.jsonl`` (optional)
    Per-call prompt metadata, raw LLM response, parsed selection, and the
    source tag returned by the gate. Enabled by ``--save_logs``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Allow running directly from the repository root without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from drktas.data_io import extract_age_from_clinical_note  # noqa: E402
from drktas.guidelines import GuidelineHelper  # noqa: E402
from drktas.prompts import load_prompt  # noqa: E402
from drktas.stage2_gate import (  # noqa: E402
    Stage2Decision,
    apply_acuity_preserving_gate,
    candidate_grade_range,
    modifier_to_level,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("drktas.stage2_infer")


# =============================================================================
# Config
# =============================================================================

@dataclass
class Stage2InferConfig:
    # Model
    backbone: str = "mistralai/Ministral-8B-Instruct-2410"
    compute_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"

    # Candidate set
    grade_expansion: int = 1
    pediatric_age_threshold: int = 15
    min_candidates: int = 2

    # Inference
    max_new_tokens: int = 512
    batch_size: int = 16
    max_seq_length: int = 4096
    seed: int = 42

    # Gate
    acuity_preserving_gate: bool = True

    # ------------------------------------------------------------------ Loaders

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Stage2InferConfig":
        cfg = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        def set_block(block: Dict[str, Any], remap: Dict[str, str]) -> None:
            for src, dst in remap.items():
                if src in block and block[src] is not None:
                    setattr(cfg, dst, block[src])

        set_block(
            data.get("model", {}),
            {
                "backbone": "backbone",
                "compute_dtype": "compute_dtype",
                "attn_implementation": "attn_implementation",
            },
        )
        set_block(
            data.get("candidate_set", {}),
            {
                "grade_expansion": "grade_expansion",
                "pediatric_age_threshold": "pediatric_age_threshold",
                "min_candidates": "min_candidates",
            },
        )
        set_block(
            data.get("data", {}),
            {"max_seq_length": "max_seq_length"},
        )
        set_block(
            data.get("inference", {}),
            {
                "max_new_tokens": "max_new_tokens",
                "batch_size": "batch_size",
                "acuity_preserving_gate": "acuity_preserving_gate",
            },
        )
        return cfg


# =============================================================================
# Stage 1 prediction parsing
# =============================================================================

def _parse_generation_string(
    raw: str, subcategory_vocabulary: set
) -> Tuple[str, str]:
    """Split a generated KTAS sequence into ``(subcategory, modifier)``.

    The Stage 1 generation head emits strings of the form
    ``"<age>, <category>, <subcategory>, <modifier>, <level>"`` (or the same
    structure without commas, depending on the trainer). This helper
    handles both. It greedily picks the longest known subcategory prefix
    from the third token onward, then takes the remainder as the modifier.
    """
    if not raw or raw == "nan":
        return "", ""

    # Strip trailing level number if present (space-separated).
    parts = raw.rsplit(" ", 1)
    text = parts[0] if len(parts) == 2 and parts[1].isdigit() else raw

    # Comma-separated layout: "<a>, <m>, <s>, <d>, <l>".
    comma_parts = [p.strip() for p in text.split(",")]
    if len(comma_parts) >= 4:
        subcategory = comma_parts[2]
        modifier = comma_parts[3]
        return subcategory, modifier

    # Whitespace-separated layout: pick the longest known subcategory prefix
    # after the first two tokens (age group + category).
    tokens = text.split()
    if len(tokens) >= 3:
        remaining = " ".join(tokens[2:])
        best_match = None
        for sub in subcategory_vocabulary:
            if remaining.startswith(sub) and (best_match is None or len(sub) > len(best_match)):
                best_match = sub
        if best_match:
            modifier = remaining[len(best_match):].strip()
            return best_match, modifier

    return "", ""


def load_stage1_predictions(
    path: str | Path, subcategory_vocabulary: set
) -> pd.DataFrame:
    """Load and normalize the Stage 1 dual-head prediction CSV."""
    df = pd.read_csv(path)

    # If the upstream CSV provides parsed fields, trust them; otherwise parse.
    if "parsed_subcategory" not in df.columns or "parsed_modifier" not in df.columns:
        if "gen_prediction" not in df.columns:
            raise ValueError(
                "Stage 1 predictions CSV must contain either 'parsed_subcategory'"
                " + 'parsed_modifier' columns, or 'gen_prediction' to parse from."
            )
        subcategories: List[str] = []
        modifiers: List[str] = []
        for raw in df["gen_prediction"].astype(str).tolist():
            s, m = _parse_generation_string(raw, subcategory_vocabulary)
            subcategories.append(s)
            modifiers.append(m)
        df["parsed_subcategory"] = subcategories
        df["parsed_modifier"] = modifiers

    # Required columns for routing.
    for col in ("gen_grade", "cls_grade"):
        if col not in df.columns:
            raise ValueError(f"Stage 1 predictions CSV is missing '{col}'.")
    if "clinical_note" not in df.columns:
        # The note is needed to extract age for pediatric routing. Fall back
        # to an empty string so the adult guideline is used by default.
        df["clinical_note"] = ""
    return df


# =============================================================================
# Prompt & response
# =============================================================================

PROMPT_TEMPLATE = load_prompt("stage2_modifier_selection")


def render_candidates(candidates: List[Dict[str, Any]]) -> str:
    return "\n".join(f'  {i + 1}. "{c["modifier"]}"' for i, c in enumerate(candidates))


_JSON_OBJECT_PATTERN = re.compile(r"\{[^{}]*\}")
_SELECTION_REGEX = re.compile(r'"선택"\s*:\s*"?(\d+)"?')


def parse_stage2_response(
    response: str, candidates: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Return ``{"selection_index", "selected_modifier", "parse_method"}``.

    ``selection_index`` is a 0-based index into ``candidates`` or ``None``
    if no valid selection could be extracted.
    """
    out: Dict[str, Any] = {
        "selection_index": None,
        "selected_modifier": None,
        "parse_method": "failed",
    }

    # 1) Try full JSON object first.
    json_match = _JSON_OBJECT_PATTERN.search(response)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if "선택" in parsed:
                selection = str(parsed["선택"]).strip()
                num_match = re.search(r"(\d+)", selection)
                if num_match:
                    idx = int(num_match.group(1)) - 1
                    if 0 <= idx < len(candidates):
                        out["selection_index"] = idx
                        out["selected_modifier"] = candidates[idx]["modifier"]
                        out["parse_method"] = "json"
                        return out
        except json.JSONDecodeError:
            pass

    # 2) Loose regex fallback on the selection key.
    m = _SELECTION_REGEX.search(response)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            out["selection_index"] = idx
            out["selected_modifier"] = candidates[idx]["modifier"]
            out["parse_method"] = "regex"
    return out


# =============================================================================
# Model
# =============================================================================

def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_model(cfg: Stage2InferConfig, lora_adapter: Optional[Path]):
    logger.info("Loading backbone: %s", cfg.backbone)
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding so batched ``generate`` produces aligned outputs.
    tokenizer.padding_side = "left"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": _torch_dtype(cfg.compute_dtype),
        "device_map": "auto",
    }
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(cfg.backbone, **model_kwargs)

    if lora_adapter is not None:
        logger.info("Loading Stage 2 LoRA adapter: %s", lora_adapter)
        model = PeftModel.from_pretrained(
            model, str(lora_adapter), torch_dtype=_torch_dtype(cfg.compute_dtype)
        )
        # Merging speeds up batched generation. The adapter weights are
        # already specialized; we don't need to keep them mutable here.
        model = model.merge_and_unload()
        logger.info("LoRA adapter merged for inference.")

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int,
    max_seq_length: int,
) -> List[str]:
    if not prompts:
        return []
    device = next(model.parameters()).device

    formatted = []
    for prompt in prompts:
        try:
            formatted.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
        except Exception:
            formatted.append(prompt)

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in outputs
    ]


# =============================================================================
# Per-case routing
# =============================================================================

@dataclass
class CaseRecord:
    """Everything we need to apply the gate after Stage 2 returns."""

    original_idx: int
    clinical_note: str
    gen_subcategory: str
    gen_modifier: str
    gen_grade: int
    cls_grade: int
    gt_grade: Optional[int]
    age: Optional[int]
    candidates: List[Dict[str, Any]]
    all_candidates: List[Dict[str, Any]]
    prompt: Optional[str]
    needs_stage2: bool
    skip_reason: Optional[str]


def build_case_records(
    df: pd.DataFrame, helper: GuidelineHelper, cfg: Stage2InferConfig
) -> List[CaseRecord]:
    records: List[CaseRecord] = []
    for idx, row in df.iterrows():
        gen_grade = int(row.get("gen_grade", 0))
        cls_grade = int(row.get("cls_grade", 0))
        gt_grade = (
            int(row["gt_grade"]) if "gt_grade" in row and pd.notna(row["gt_grade"]) else None
        )
        clinical_note = str(row.get("clinical_note", "") or "")
        gen_subcategory = str(row.get("parsed_subcategory", "") or "")
        gen_modifier = str(row.get("parsed_modifier", "") or "")

        age = extract_age_from_clinical_note(clinical_note)
        candidates = helper.get_constrained_modifier_candidates(
            subcategory=gen_subcategory,
            gen_level=gen_grade,
            cls_level=cls_grade,
            age=age,
            grade_expansion=cfg.grade_expansion,
        )
        all_candidates = helper.get_modifier_candidates(gen_subcategory, age=age)

        if not gen_subcategory:
            record = CaseRecord(
                original_idx=int(idx),
                clinical_note=clinical_note,
                gen_subcategory=gen_subcategory,
                gen_modifier=gen_modifier,
                gen_grade=gen_grade,
                cls_grade=cls_grade,
                gt_grade=gt_grade,
                age=age,
                candidates=[],
                all_candidates=all_candidates,
                prompt=None,
                needs_stage2=False,
                skip_reason="no_subcategory",
            )
        elif len(candidates) < cfg.min_candidates:
            record = CaseRecord(
                original_idx=int(idx),
                clinical_note=clinical_note,
                gen_subcategory=gen_subcategory,
                gen_modifier=gen_modifier,
                gen_grade=gen_grade,
                cls_grade=cls_grade,
                gt_grade=gt_grade,
                age=age,
                candidates=candidates,
                all_candidates=all_candidates,
                prompt=None,
                needs_stage2=False,
                skip_reason="insufficient_candidates",
            )
        else:
            prompt = PROMPT_TEMPLATE.format(
                clinical_note=clinical_note,
                candidates_text=render_candidates(candidates),
            )
            record = CaseRecord(
                original_idx=int(idx),
                clinical_note=clinical_note,
                gen_subcategory=gen_subcategory,
                gen_modifier=gen_modifier,
                gen_grade=gen_grade,
                cls_grade=cls_grade,
                gt_grade=gt_grade,
                age=age,
                candidates=candidates,
                all_candidates=all_candidates,
                prompt=prompt,
                needs_stage2=True,
                skip_reason=None,
            )
        records.append(record)
    return records


def run_inference(
    records: List[CaseRecord],
    model,
    tokenizer,
    cfg: Stage2InferConfig,
    log_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Apply Stage 2 generation + gate to every record."""
    routed = [r for r in records if r.needs_stage2]
    results_by_idx: Dict[int, Dict[str, Any]] = {}

    log_writer = log_path.open("w", encoding="utf-8") if log_path is not None else None
    try:
        # Apply the gate to head-agreement cases or skipped cases first.
        for record in records:
            if not record.needs_stage2:
                decision = apply_acuity_preserving_gate(
                    gen_level=record.gen_grade,
                    cls_level=record.cls_grade,
                    stage2_level=None,
                    stage2_modifier=None,
                    gen_modifier=record.gen_modifier,
                    has_sufficient_candidates=len(record.candidates) >= cfg.min_candidates,
                )
                results_by_idx[record.original_idx] = _decision_to_row(
                    record, decision, stage2_response=None, parse_method=record.skip_reason or "no_routing"
                )

        # Batched Stage 2 calls for head-disagreement cases.
        for start in tqdm(range(0, len(routed), cfg.batch_size), desc="Stage 2"):
            batch = routed[start : start + cfg.batch_size]
            prompts = [r.prompt for r in batch]
            responses = generate_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens=cfg.max_new_tokens,
                max_seq_length=cfg.max_seq_length,
            )

            for record, response in zip(batch, responses):
                parsed = parse_stage2_response(response, record.candidates)
                if parsed["selection_index"] is None:
                    stage2_level = None
                    stage2_modifier = None
                else:
                    stage2_level = modifier_to_level(
                        parsed["selected_modifier"], record.candidates
                    )
                    stage2_modifier = parsed["selected_modifier"]

                decision = apply_acuity_preserving_gate(
                    gen_level=record.gen_grade,
                    cls_level=record.cls_grade,
                    stage2_level=stage2_level,
                    stage2_modifier=stage2_modifier,
                    gen_modifier=record.gen_modifier,
                    has_sufficient_candidates=True,
                )
                if not cfg.acuity_preserving_gate and stage2_level is not None:
                    # Bypass the gate (sensitivity ablation): take Stage 2 directly.
                    decision = Stage2Decision(
                        final_level=int(stage2_level),
                        final_modifier=stage2_modifier,
                        source="stage2_accept_ungated",
                    )

                row = _decision_to_row(
                    record,
                    decision,
                    stage2_response=response,
                    parse_method=parsed["parse_method"],
                )
                row["stage2_level"] = stage2_level
                row["stage2_modifier"] = stage2_modifier
                row["selection_index"] = parsed["selection_index"]
                results_by_idx[record.original_idx] = row

                if log_writer is not None:
                    log_writer.write(
                        json.dumps(
                            {
                                "original_idx": record.original_idx,
                                "age": record.age,
                                "gen_grade": record.gen_grade,
                                "cls_grade": record.cls_grade,
                                "gt_grade": record.gt_grade,
                                "gen_subcategory": record.gen_subcategory,
                                "gen_modifier": record.gen_modifier,
                                "candidates": record.candidates,
                                "stage2_level": stage2_level,
                                "stage2_modifier": stage2_modifier,
                                "parse_method": parsed["parse_method"],
                                "final_level": decision.final_level,
                                "source": decision.source,
                                "raw_response": response,
                                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    finally:
        if log_writer is not None:
            log_writer.close()

    return [results_by_idx[r.original_idx] for r in records]


def _decision_to_row(
    record: CaseRecord,
    decision: Stage2Decision,
    *,
    stage2_response: Optional[str],
    parse_method: Optional[str],
) -> Dict[str, Any]:
    return {
        "original_idx": record.original_idx,
        "gt_grade": record.gt_grade,
        "gen_grade": record.gen_grade,
        "cls_grade": record.cls_grade,
        "gen_subcategory": record.gen_subcategory,
        "gen_modifier": record.gen_modifier,
        "age": record.age,
        "num_candidates": len(record.candidates),
        "num_all_candidates": len(record.all_candidates),
        "final_level": int(decision.final_level),
        "final_modifier": decision.final_modifier,
        "source": decision.source,
        "stage2_level": None,
        "stage2_modifier": None,
        "selection_index": None,
        "parse_method": parse_method,
        "raw_response": stage2_response,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 inference for Dr.KTAS")
    parser.add_argument("--config", type=str, required=True, help="YAML config")
    parser.add_argument("--stage1_predictions", type=str, required=True)
    parser.add_argument("--lora_adapter", type=str, default=None)
    parser.add_argument(
        "--guidelines_dir",
        type=str,
        default=str(_REPO_ROOT / "guidelines"),
        help="Directory containing ktas_{adult,children}_guideline_lookup_clean.json",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Optional cap on the number of records evaluated.",
    )
    parser.add_argument(
        "--save_logs",
        action="store_true",
        help="Write verifier_logs.jsonl with per-case prompts and raw responses.",
    )
    return parser.parse_args()


def routing_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sources = Counter(row["source"] for row in rows)
    return {
        "n_total": len(rows),
        "n_routed": sum(
            sources[k]
            for k in ("stage2_accept", "stage2_reject_gate", "stage2_invalid", "stage2_accept_ungated")
        ),
        "by_source": dict(sources),
    }


def main() -> None:
    args = parse_args()
    cfg = Stage2InferConfig.from_yaml(args.config)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    guidelines_dir = Path(args.guidelines_dir)
    helper = GuidelineHelper(
        guidelines_dir / "ktas_adult_guideline_lookup_clean.json",
        guidelines_dir / "ktas_children_guideline_lookup_clean.json",
    )
    logger.info(
        "Guideline loaded: adult=%d children=%d subcategories",
        len(helper.adult_subcategory_to_modifiers),
        len(helper.children_subcategory_to_modifiers),
    )

    df = load_stage1_predictions(args.stage1_predictions, helper.subcategory_vocabulary)
    if args.num_samples is not None:
        df = df.head(args.num_samples)
    logger.info(
        "Loaded %d Stage 1 predictions; head-disagreement = %d",
        len(df),
        int((df["gen_grade"] != df["cls_grade"]).sum()),
    )

    records = build_case_records(df, helper, cfg)

    model, tokenizer = load_model(cfg, lora_adapter=Path(args.lora_adapter) if args.lora_adapter else None)

    log_path = output_dir / "verifier_logs.jsonl" if args.save_logs else None
    rows = run_inference(records, model, tokenizer, cfg, log_path=log_path)

    pred_df = pd.DataFrame(rows)
    pred_path = output_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info("Saved per-case predictions to %s", pred_path)

    summary = routing_summary(rows)
    (output_dir / "routing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    logger.info(
        "Routing summary: %s",
        json.dumps(summary, ensure_ascii=False),
    )


if __name__ == "__main__":
    main()
