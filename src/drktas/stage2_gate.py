"""Stage 2 acuity-preserving gate.

The Dr.KTAS pipeline accepts a Stage 2 (Guideline-Informed Modifier
Re-adjudication) selection only when it preserves or increases acuity
relative to the Stage 1 generation output:

``hat_ell_DrKTAS = hat_ell_S2  if routed, valid and hat_ell_S2 <= hat_ell_gen``
``                hat_ell_gen otherwise.``

This module collects the small pieces of decision logic that surround the
LLM call: deciding whether to route at all, choosing the candidate range,
converting a selected modifier back to a KTAS level, and applying the
acceptance rule. The reasoning behind it is documented in the paper
(Section III-E).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Stage2Decision:
    """Final routing outcome for a single case.

    ``final_level`` is the level returned to the caller. ``source`` is one
    of ``'stage1_gen'``, ``'stage2_accept'``, ``'stage2_reject_gate'``,
    ``'stage2_invalid'``, ``'no_routing'`` or ``'no_candidates'`` and
    captures why the value was chosen, which lets the bootstrap CI scripts
    classify routed-case behavior.
    """

    final_level: int
    final_modifier: Optional[str]
    source: str


def should_route_to_stage2(gen_level: int, cls_level: int) -> bool:
    """Return True when the two Stage 1 heads disagree on the level."""
    return int(gen_level) != int(cls_level)


def candidate_grade_range(
    gen_level: int,
    cls_level: int,
    expansion: int = 1,
    min_level: int = 1,
    max_level: int = 5,
) -> Tuple[int, int]:
    """Boundary-expanded grade range for Stage 2 candidate retrieval.

    Implements equation (5) of the paper. With ``expansion = 1`` the range
    extends one level on either side of the interval spanned by the two
    Stage 1 predictions, clipped to ``[min_level, max_level]``.
    """
    lo = max(min_level, min(gen_level, cls_level) - expansion)
    hi = min(max_level, max(gen_level, cls_level) + expansion)
    return lo, hi


def modifier_to_level(
    selected_modifier: str,
    candidates: List[Dict[str, object]],
) -> Optional[int]:
    """Look up the KTAS level that a selected modifier maps to.

    ``candidates`` is the list returned by
    :func:`drktas.guidelines.GuidelineHelper.get_modifier_candidates`; each
    entry has a ``"modifier"`` and a ``"level"`` field. Returns ``None``
    when the selection is not present in the candidate list.
    """
    for entry in candidates:
        if entry.get("modifier") == selected_modifier:
            try:
                return int(entry["level"])
            except (TypeError, ValueError):
                return None
    return None


def apply_acuity_preserving_gate(
    gen_level: int,
    cls_level: int,
    stage2_level: Optional[int],
    stage2_modifier: Optional[str],
    gen_modifier: Optional[str],
    has_sufficient_candidates: bool,
) -> Stage2Decision:
    """Apply the Stage 2 acceptance rule and return the chosen level.

    Parameters
    ----------
    gen_level, cls_level:
        The Stage 1 generation and classification head predictions
        (integers in ``[1, 5]``).
    stage2_level:
        The level implied by the Stage 2 selection (via
        :func:`modifier_to_level`). ``None`` indicates an invalid parse.
    stage2_modifier:
        The modifier string Stage 2 selected, if any.
    gen_modifier:
        The modifier produced by the Stage 1 generation head; used as the
        fallback modifier when Stage 2 is rejected.
    has_sufficient_candidates:
        ``True`` when at least two guideline candidates were retrieved for
        Stage 2 to choose between. The paper requires at least two
        candidates for routing.
    """
    if not should_route_to_stage2(gen_level, cls_level):
        return Stage2Decision(
            final_level=int(gen_level),
            final_modifier=gen_modifier,
            source="no_routing",
        )

    if not has_sufficient_candidates:
        return Stage2Decision(
            final_level=int(gen_level),
            final_modifier=gen_modifier,
            source="no_candidates",
        )

    if stage2_level is None:
        return Stage2Decision(
            final_level=int(gen_level),
            final_modifier=gen_modifier,
            source="stage2_invalid",
        )

    # Lower numbers indicate higher acuity, so the gate accepts only
    # selections that are at least as urgent as the Stage 1 generation
    # output.
    if int(stage2_level) <= int(gen_level):
        return Stage2Decision(
            final_level=int(stage2_level),
            final_modifier=stage2_modifier,
            source="stage2_accept",
        )
    return Stage2Decision(
        final_level=int(gen_level),
        final_modifier=gen_modifier,
        source="stage2_reject_gate",
    )
