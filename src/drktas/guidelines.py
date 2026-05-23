"""KTAS guideline lookup and candidate retrieval.

The KTAS guideline encodes a deterministic mapping from
``(chief-complaint category, subcategory, clinical modifier)`` to a final
KTAS level. This module loads the adult and pediatric lookup tables, exposes
the modifier candidates for a given subcategory, and implements the
boundary-aware candidate-range filter used by Stage 2 re-adjudication.

The lookup JSON files distributed in ``guidelines/`` use the field names
introduced by the institutional curation pipeline:

* ``Lv3exp``   — subcategory description (chief-complaint subcategory ``s``)
* ``Lv4exp``   — clinical modifier description (``d``)
* ``severity`` — KTAS level (1-5) for the modifier (``l``)

The file is read with :mod:`ftfy` to fix legacy encoding artifacts inherited
from the source records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import ftfy


PEDIATRIC_AGE_THRESHOLD = 15  # Below this age the pediatric guideline applies.


class GuidelineHelper:
    """Access adult and pediatric KTAS lookup tables.

    Parameters
    ----------
    adult_path:
        Path to ``ktas_adult_guideline_lookup_clean.json``.
    children_path:
        Path to ``ktas_children_guideline_lookup_clean.json``.
    """

    def __init__(self, adult_path: str | Path, children_path: str | Path) -> None:
        self.adult_guideline = self._load_lookup(adult_path)
        self.children_guideline = self._load_lookup(children_path)

        self.adult_subcategory_to_modifiers = self._build_subcategory_mapping(
            self.adult_guideline
        )
        self.children_subcategory_to_modifiers = self._build_subcategory_mapping(
            self.children_guideline
        )

        self.subcategory_vocabulary = set(self.adult_subcategory_to_modifiers) | set(
            self.children_subcategory_to_modifiers
        )

    # ------------------------------------------------------------------ I/O

    @staticmethod
    def _load_lookup(path: str | Path) -> Dict:
        """Load a guideline JSON file, repairing encoding artifacts."""
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        return json.loads(ftfy.fix_text(raw))

    @staticmethod
    def _build_subcategory_mapping(
        guideline: Dict,
    ) -> Dict[str, List[Dict[str, object]]]:
        """Index modifiers by subcategory, sorted by KTAS level (ascending)."""
        mapping: Dict[str, List[Dict[str, object]]] = {}
        for entry in guideline.values():
            subcategory = entry.get("Lv3exp", "")
            modifier = entry.get("Lv4exp", "")
            level = entry.get("severity", 0)
            if not subcategory:
                continue

            bucket = mapping.setdefault(subcategory, [])
            already_seen = {item["modifier"] for item in bucket}
            if modifier not in already_seen:
                bucket.append({"modifier": modifier, "level": level})

        for key in mapping:
            mapping[key].sort(key=lambda item: item["level"])
        return mapping

    # ------------------------------------------------------------------ Queries

    @staticmethod
    def is_pediatric(age: Optional[int]) -> bool:
        """Return True if the patient should be routed to the pediatric guideline.

        Falls back to ``False`` when the age cannot be parsed, matching the
        institutional rule of defaulting to the adult guideline on ambiguity.
        """
        if age is None:
            return False
        return age < PEDIATRIC_AGE_THRESHOLD

    def get_modifier_candidates(
        self,
        subcategory: str,
        age: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        """Return all modifier candidates for a subcategory.

        If the age-appropriate guideline does not list the subcategory, the
        other guideline is consulted as a fallback so that retrieval never
        fails purely on the basis of age routing.
        """
        if self.is_pediatric(age):
            primary = self.children_subcategory_to_modifiers
            fallback = self.adult_subcategory_to_modifiers
        else:
            primary = self.adult_subcategory_to_modifiers
            fallback = self.children_subcategory_to_modifiers

        candidates = primary.get(subcategory, [])
        if not candidates:
            candidates = fallback.get(subcategory, [])
        return candidates

    def get_constrained_modifier_candidates(
        self,
        subcategory: str,
        gen_level: int,
        cls_level: int,
        age: Optional[int] = None,
        grade_expansion: int = 1,
    ) -> List[Dict[str, object]]:
        """Return modifier candidates whose KTAS levels fall within
        the boundary-expanded range used by Stage 2.

        The accepted range is
        ``[max(1, min(gen, cls) - expansion), min(5, max(gen, cls) + expansion)]``,
        matching equation (5) in the paper with ``expansion = 1``.
        """
        all_candidates = self.get_modifier_candidates(subcategory, age)
        lo = max(1, min(gen_level, cls_level) - grade_expansion)
        hi = min(5, max(gen_level, cls_level) + grade_expansion)
        return [c for c in all_candidates if lo <= int(c["level"]) <= hi]
