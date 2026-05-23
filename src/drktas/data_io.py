"""I/O helpers for the KTAS pipeline.

This module covers the parts of the data path that are shared across the
Stage 1 trainer, the Stage 2 data preparation script, the Stage 2 inference
pipeline, and the baseline inference scripts:

* A delimiter constant separating the prompt from the KTAS response.
* Robust extraction of the age, KTAS level, and adjudication-sequence tokens
  from the institutionally curated ``fullseverity`` field.
* Inverse-frequency class weight computation for the Stage 1 classification
  head, following equation (4) in the paper.

The free-text patient prompt itself is constructed upstream of this code and
passed in as the ``text`` column of the CSV. The trainer simply concatenates
``text + response + tokenizer.eos_token`` and masks the prompt tokens with
``-100``.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


KTAS_SEQUENCE_DELIMITER = "[KTAS sequence]"
"""Token boundary placed at the end of the prompt.

The Stage 1 classification head reads the hidden state at the position of
the last prompt token, which is the final token of this delimiter, so the
backbone has seen the entire clinical record before producing the
classification logits.
"""


# ---------------------------------------------------------------------- Parsing

_AGE_PATTERNS: Tuple[str, ...] = (
    r"-\s*나이[:\s]+(\d+)",
    r"나이[:\s]+(\d+)",
    r"age[:\s]+(\d+)",
    r"(\d+)\s*세",
    r"(\d+)\s*살",
    r"(\d+)\s*y/?o",
)


def extract_age_from_clinical_note(text: str) -> Optional[int]:
    """Return the age (years) parsed from a Korean/English clinical note.

    Returns ``None`` when no age pattern is found, in which case downstream
    code routes to the adult guideline by default.
    """
    if not text:
        return None
    for pattern in _AGE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue
    return None


def extract_level_from_fullseverity(fullseverity: str) -> Optional[int]:
    """Extract the KTAS level (1-5) from the comma-separated sequence.

    The institutional field stores the sequence as
    ``"<age group>, <category>, <subcategory>, <modifier>, <level>"`` with
    the level as the trailing token.
    """
    if not fullseverity:
        return None

    text = fullseverity.strip()
    if not text:
        return None

    last_token = text.split()[-1]
    try:
        level = int(last_token)
        if 1 <= level <= 5:
            return level
    except ValueError:
        pass

    match = re.search(r"(\d)$", text)
    if match:
        level = int(match.group(1))
        if 1 <= level <= 5:
            return level
    return None


def split_fullseverity_tokens(fullseverity: str) -> Optional[Tuple[str, str, str, str, int]]:
    """Split a ``fullseverity`` string into ``(a, m, s, d, l)``.

    Returns ``None`` when the field does not contain all five components.
    The implementation is tolerant of whitespace around the commas but does
    not attempt to reconstruct sequences that do not follow the documented
    schema.
    """
    if not fullseverity:
        return None
    parts = [part.strip() for part in fullseverity.split(",")]
    if len(parts) != 5:
        return None
    try:
        level = int(parts[-1])
    except ValueError:
        return None
    if not (1 <= level <= 5):
        return None
    return parts[0], parts[1], parts[2], parts[3], level


# ----------------------------------------------------- Class weights (Eq. 4)


def compute_inverse_frequency_weights(
    labels: Iterable[int],
    num_classes: int = 5,
    gamma: float = 1.0,
) -> np.ndarray:
    """Inverse-frequency class weights with unit-mean normalization.

    Implements equation (4) in the paper:

    ``w_i = (N / (K * N_i)) ** gamma``
    then rescaled so that the arithmetic mean of the weights is one.

    Parameters
    ----------
    labels:
        Iterable of integer class labels in ``[1, num_classes]`` *or*
        ``[0, num_classes - 1]``. The function infers the offset from the
        observed minimum.
    num_classes:
        Number of classes ``K`` (default 5 for KTAS levels).
    gamma:
        Inverse-frequency exponent. ``gamma = 1.0`` is the paper's default
        and corresponds to standard inverse-frequency reweighting.

    Returns
    -------
    weights:
        Array of shape ``(num_classes,)`` with unit mean.
    """
    label_array = np.asarray(list(labels), dtype=np.int64)
    if label_array.size == 0:
        return np.ones(num_classes, dtype=np.float64)

    # Allow both 0- and 1-indexed labels.
    offset = 1 if label_array.min() >= 1 else 0
    counts = np.zeros(num_classes, dtype=np.int64)
    for value in label_array:
        idx = int(value) - offset
        if 0 <= idx < num_classes:
            counts[idx] += 1

    total = counts.sum()
    if total == 0:
        return np.ones(num_classes, dtype=np.float64)

    safe_counts = np.where(counts > 0, counts, 1)
    raw = (total / (num_classes * safe_counts)) ** gamma
    # Classes that never appear should not be down-weighted to zero; they
    # keep the raw value, which is large because counts were clamped to 1.
    weights = raw / raw.mean()
    return weights.astype(np.float64)


# --------------------------------------------------------------- Prompt format


def build_response_text(fullseverity: str, target_format: str) -> str:
    """Return the response string supervised by the generation head.

    Parameters
    ----------
    fullseverity:
        The documented KTAS adjudication sequence stored in the CSV.
    target_format:
        Either ``'full_sequence'`` to supervise the entire adjudication
        tuple (Triage-full context, Dual-Head), or ``'level'`` to supervise
        only the final KTAS level digit (Triage-level ablation).

    Raises
    ------
    ValueError:
        On unknown ``target_format`` or unparseable ``fullseverity``.
    """
    if target_format == "full_sequence":
        return fullseverity.strip()
    if target_format == "level":
        level = extract_level_from_fullseverity(fullseverity)
        if level is None:
            raise ValueError(
                "Cannot extract a KTAS level (1-5) from fullseverity="
                f"{fullseverity!r}; required for target_format='level'."
            )
        return str(level)
    raise ValueError(
        f"Unknown target_format={target_format!r}. "
        "Use 'full_sequence' or 'level'."
    )


def class_weights_for_loader(
    labels: Sequence[int],
    num_classes: int = 5,
    gamma: float = 1.0,
) -> List[float]:
    """Convenience wrapper that returns a plain Python list.

    Useful for the YAML-driven configs that record the weights for logging.
    """
    return compute_inverse_frequency_weights(labels, num_classes, gamma).tolist()
