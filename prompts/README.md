# Prompts

Single source of truth for every prompt template the pipeline uses. Each
file in this directory is loaded at run-time by the scripts that need it,
so updating a prompt here updates the corresponding code path without
further edits.

The prompts retain their original Korean wording because the Stage 2 LoRA
adapter and the trained baselines were optimized for these exact strings;
changing the natural-language content invalidates the trained weights.

## Files

| File | Used by | Placeholders |
|---|---|---|
| `ktas_grade_info.txt` | `scripts/baseline_infer.py` | none |
| `baseline_no_description.txt` | `scripts/baseline_infer.py` (prompt_type=`no_description`) | none (pass-through) |
| `baseline_with_description.txt` | `scripts/baseline_infer.py` (prompt_type=`with_description`) | `{patient_info}`, `{grade_info}` |
| `baseline_constrained.txt` | `scripts/baseline_infer.py` (prompt_type=`constrained`) | `{patient_info}`, `{grade_info}` |
| `stage2_modifier_selection.txt` | `scripts/stage2_infer.py`, `scripts/stage2_prepare_data.py` | `{clinical_note}`, `{candidates_text}` |

## Usage from Python

Prompts are loaded with :func:`drktas.prompts.load_prompt`:

```python
from drktas.prompts import load_prompt
template = load_prompt("stage2_modifier_selection")
rendered = template.format(clinical_note=..., candidates_text=...)
```

The loader uses Python's :py:meth:`str.format`, so any literal ``{`` or
``}`` characters in the prompt body must be escaped as ``{{`` / ``}}``.

## Editing

When changing a prompt, keep the placeholder names identical so existing
scripts continue to format correctly. If you add a new placeholder,
update both the template file and the call sites that render it.

If you are modifying a prompt that was used to train the released Stage 2
LoRA adapter, expect to retrain the adapter before using the new wording
at inference time. The candidate-number JSON schema (`"검증결과"`,
`"선택"`, `"이유"`) and the surrounding key markers must be preserved
unless you also update the corresponding parsing logic and loss-masking
markers in the Stage 2 trainer and inference scripts.
