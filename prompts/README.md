# Prompts

This directory contains six fully rendered prompt examples aligned with the
submitted Dr.KTAS manuscript and Fig. 2. They are provided so readers can inspect
the exact input style used for the model variants and zero-shot comparators.

No real patient record is included. The example record is a synthetic,
illustrative clinical note and must not be used for clinical decision-making or
benchmarking.

## Files

| File | Purpose |
|---|---|
| `final_level_only_prompt.txt` | Example structured triage-time prompt for the Final-level only model. |
| `complete_sequence_prompt.txt` | Example structured triage-time prompt for the Complete sequence model. |
| `classification_only_prompt.txt` | Example structured triage-time prompt for the Classification-only model. |
| `dual_head_prompt.txt` | Example structured triage-time prompt for the Dual-Head model. |
| `zero_shot_with_description_prompt.txt` | Fully rendered zero-shot final-level KTAS prompt with KTAS level descriptions, used to document the prompt style for Ministral-8B, Meditron3-14B, and GPT-4o. |
| `Dr.KTAS_prompt.txt` | Fully rendered Dr.KTAS example showing the clinical record, Stage 2 modifier candidates, verification instructions, and output format. |

## Model Variant Prompts

The four Stage 1 model examples use the same structured triage-time clinical
prompt. This reflects the controlled comparison in the manuscript: model
variants differ in supervision target and model head design, not in the clinical
information shown in the input prompt.

The structured KTAS reference fields are not included in the input prompt. They
are used as supervised targets and evaluation references in the paper.

## Zero-Shot Comparator Prompt

`zero_shot_with_description_prompt.txt` shows the prompt style used for the
zero-shot LLM comparators. It contains the same clinical record followed by KTAS
level descriptions and asks the model to output only the final KTAS level.

## Dr.KTAS Prompt

`Dr.KTAS_prompt.txt` illustrates the Dr.KTAS workflow after Stage 1 routing. It
uses the same clinical record and appends a bounded list of KTAS-compatible
clinical modifier candidates. The prompt asks the model to select the candidate
number that best reflects the patient record.

The non-deescalation gate described in the manuscript is applied after parsing
the Stage 2 output; it is not a free-form prompt instruction.
