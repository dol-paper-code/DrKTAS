"""Loader for the prompt templates under ``prompts/``.

This module exists so that the prompt wording lives in a single
human-readable place. Callers obtain a template string via
:func:`load_prompt` and format it with the appropriate placeholders.

The loader caches results so repeatedly reading the same template within
a process is free, and it tolerates running the scripts from arbitrary
working directories by searching up the source tree for the
``prompts/`` folder.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional


_PROMPT_FILES = {
    "final_level_only_prompt": "final_level_only_prompt.txt",
    "complete_sequence_prompt": "complete_sequence_prompt.txt",
    "classification_only_prompt": "classification_only_prompt.txt",
    "dual_head_prompt": "dual_head_prompt.txt",
    "drktas_prompt": "Dr.KTAS_prompt.txt",
    "zero_shot_with_description_prompt": "zero_shot_with_description_prompt.txt",
}


def _candidate_roots() -> Iterable[Path]:
    """Yield directories where ``prompts/`` could live.

    The expected layout when running from the repository root or scripts/
    folder is ``<repo_root>/prompts/``, which is two parents above this
    file. We also walk up the parent chain so callers running from
    nested directories still resolve the folder.
    """
    here = Path(__file__).resolve()
    # repo_root/src/drktas/prompts.py -> repo_root/prompts
    yield here.parents[2] / "prompts"
    # Walk further up in case the package is installed inside a wheel
    # alongside a sibling prompts/ folder.
    for parent in here.parents:
        candidate = parent / "prompts"
        if candidate.is_dir():
            yield candidate
    # As a last resort honor the current working directory.
    yield Path.cwd() / "prompts"


def prompts_dir(override: Optional[Path] = None) -> Path:
    """Return the resolved prompts directory.

    Pass ``override`` to point the loader at an alternate prompt set; the
    function still verifies that the directory exists.
    """
    if override is not None:
        path = Path(override)
        if not path.is_dir():
            raise FileNotFoundError(f"Prompts directory not found: {path}")
        return path
    for candidate in _candidate_roots():
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the prompts/ directory. Set DRKTAS_PROMPTS_DIR or "
        "pass an explicit path to load_prompt()."
    )


@lru_cache(maxsize=None)
def _read(name: str, root: Path) -> str:
    file_name = _PROMPT_FILES[name]
    text = (root / file_name).read_text(encoding="utf-8")
    # Allow trailing newlines without forcing them onto every call site.
    return text.rstrip("\n")


def load_prompt(name: str, *, prompts_root: Optional[Path] = None) -> str:
    """Return the raw template text for ``name``.

    Parameters
    ----------
    name:
        One of the keys in :data:`_PROMPT_FILES`.
    prompts_root:
        Optional override directory.

    Raises
    ------
    KeyError:
        If ``name`` is not a registered template.
    """
    if name not in _PROMPT_FILES:
        raise KeyError(
            f"Unknown prompt {name!r}. Known prompts: {sorted(_PROMPT_FILES)}"
        )
    return _read(name, prompts_dir(prompts_root))


def list_prompts() -> list[str]:
    """Return the names of every registered prompt."""
    return sorted(_PROMPT_FILES)
