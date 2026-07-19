"""Run directory layout and typed artifact I/O.

One RunDir == one idea validation run. All stage artifacts live under root
(design §3). Adapted from paper_reprise.rundir.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

from spike_eval.models import Grade, IdeaSpec


def slug(text: str, max_len: int = 40) -> str:
    """Filesystem-safe slug. Uses the part before the first colon when non-empty
    (idea names are often "ShortName: subtitle")."""
    head = text.split(":", 1)[0]
    s = re.sub(r"[^a-z0-9]+", "-", head.lower()).strip("-")
    if not s:
        s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].strip("-") or "idea"


class RunDir:
    """Typed I/O over one run directory. Directory creation is eager; artifact writes
    are per-stage."""

    def __init__(self, root: Path):
        self.root = Path(root)

    # ---- lifecycle -------------------------------------------------------

    @classmethod
    def create(cls, base: Path, idea_name: str, repo_name: str, timestamp: str) -> "RunDir":
        name = f"{slug(idea_name)}-{slug(repo_name)}-{timestamp}"
        root = Path(base) / name
        rd = cls(root)
        for d in (rd.root, rd.repo_dir, rd.env_dir, rd.baseline_dir, rd.impl_dir,
                  rd.correctness_dir, rd.ladder_dir, rd.ablation_dir):
            d.mkdir(parents=True, exist_ok=True)
        return rd

    @classmethod
    def open(cls, root: Path) -> "RunDir":
        rd = cls(Path(root))
        if not rd.root.is_dir():
            raise FileNotFoundError(f"run dir not found: {rd.root}")
        return rd

    # ---- paths -----------------------------------------------------------

    @property
    def repo_dir(self) -> Path:
        return self.root / "repo"

    @property
    def env_dir(self) -> Path:
        return self.root / "env"

    @property
    def baseline_dir(self) -> Path:
        return self.root / "baseline"

    @property
    def impl_dir(self) -> Path:
        return self.root / "impl"

    @property
    def correctness_dir(self) -> Path:
        return self.root / "correctness"

    @property
    def ladder_dir(self) -> Path:
        return self.root / "ladder"

    @property
    def ablation_dir(self) -> Path:
        return self.root / "ablation"

    # ---- typed artifact I/O ---------------------------------------------

    def write_idea(self, text: str) -> None:
        (self.root / "idea.md").write_text(text)

    def read_idea(self) -> Optional[str]:
        p = self.root / "idea.md"
        return p.read_text() if p.exists() else None

    def write_spec(self, spec: IdeaSpec) -> None:
        (self.root / "idea_spec.yaml").write_text(
            yaml.safe_dump(spec.model_dump(), sort_keys=False, allow_unicode=True))

    def read_spec(self) -> Optional[IdeaSpec]:
        p = self.root / "idea_spec.yaml"
        if not p.exists():
            return None
        return IdeaSpec.model_validate(yaml.safe_load(p.read_text()))

    def write_grade(self, grade: Grade) -> None:
        (self.root / "grade.json").write_text(grade.model_dump_json(indent=2))

    def read_grade(self) -> Optional[Grade]:
        p = self.root / "grade.json"
        if not p.exists():
            return None
        return Grade.model_validate_json(p.read_text())

    def repo_present(self) -> bool:
        """True iff a repo was cloned/copied into repo_dir (the dir is always mkdir-ed,
        so test non-emptiness)."""
        return self.repo_dir.is_dir() and any(self.repo_dir.iterdir())
