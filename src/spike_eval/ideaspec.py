"""ideaspec — idea + repo -> falsifiable IdeaSpec (design §2.1, gate 1).

The extraction itself is agentic (headless Claude reads the idea + the repo, finds the
cleanest extension point, and writes a machine-checkable spec). Scope A ships the
interface + a pure parser/validator; scope B wires the headless call.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import yaml

from spike_eval.models import IdeaSpec
from spike_eval.rundir import RunDir

# Executor contract: given the run dir (repo/ + idea.md present), produce idea_spec.yaml
# and return the parsed IdeaSpec (or None on failure). Injected so the pipeline is
# offline-testable; the real one drives headless Claude.
SpecExtractor = Callable[[RunDir], Optional[IdeaSpec]]


def parse_spec_file(path: Path) -> IdeaSpec:
    """Pure: load + validate an idea_spec.yaml written by the extractor (or hand-edited
    by the user at gate 1). Raises pydantic ValidationError on a bad spec."""
    return IdeaSpec.model_validate(yaml.safe_load(Path(path).read_text()))


def extract_spec(rd: RunDir, extractor: Optional[SpecExtractor]) -> Optional[IdeaSpec]:
    """Run the injected extractor, or raise if none supplied (scope B fills the default
    headless extractor). If idea_spec.yaml already exists (resume / hand-authored), it is
    loaded and validated instead of re-extracted."""
    existing = rd.root / "idea_spec.yaml"
    if existing.exists():
        return parse_spec_file(existing)
    if extractor is None:
        raise NotImplementedError(
            "ideaspec extractor not wired (scope B). Provide idea_spec.yaml in the run "
            "dir to resume, or inject a SpecExtractor.")
    return extractor(rd)
