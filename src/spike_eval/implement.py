"""implement — headless Claude writes a surgical diff at the extension point (design §2.1).

Works on an isolated copy (rd.repo_dir) so the baseline is never mutated in place; the
resulting patch is persisted to rd.impl_dir. Agentic; scope A ships the interface only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from spike_eval.models import IdeaSpec
from spike_eval.rundir import RunDir


@dataclass
class ImplementResult:
    ok: bool
    patch_path: Optional[str] = None      # rd.impl_dir/idea.patch
    notes: str = ""
    files_touched: list[str] = field(default_factory=list)


# Executor contract: given the run dir + spec (redacted view is written by the pipeline),
# produce the surgical diff and return an ImplementResult. Injected for offline tests.
Implementer = Callable[[RunDir, IdeaSpec], ImplementResult]


def run_implement(rd: RunDir, spec: IdeaSpec, implementer: Optional[Implementer]
                  ) -> ImplementResult:
    if implementer is None:
        raise NotImplementedError("implementer not wired (scope B)")
    return implementer(rd, spec)
