"""Typed contracts shared across all pipeline stages.

This module depends on nothing else in spike_eval — it is the pure schema. Mirrors
paper_reprise.models in spirit, but the domain is "does this idea beat the repo's own
baseline?" rather than "does this match the paper's number?".
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# --- enums -----------------------------------------------------------------

# What the idea claims to win on. Drives what the ladder measures.
WinsOn = Literal["accuracy", "bits", "speed", "memory", "other"]

# Where the idea is grafted into the repo. Lower blast radius is preferred so the
# baseline stays untouched and the diff is reversible.
ExtKind = Literal["subclass", "register", "hook", "core-edit"]

# Kinds of correctness check, cheapest/strongest first.
CheckKind = Literal["unit", "closed_form", "degenerate_equivalence"]

# Cost-ladder tiers, cheapest first (L0 correctness is a separate hard gate).
TierName = Literal["L1_proxy", "L2_tiny", "L3_small", "L4_full"]

# Pure-code go/no-go verdict.
Verdict = Literal["WIN", "NEUTRAL", "LOSE", "BLOCKED"]


# --- eval protocol ---------------------------------------------------------


class EvalProtocol(BaseModel):
    """How a metric is measured, fixed identically for baseline and idea so the
    comparison is fair."""
    command: str                      # the repo's own eval command, when available
    metric: str                       # "perplexity" | "accuracy" | "mse" | "speedup" | ...
    lower_is_better: bool = True      # ppl/mse: True ; accuracy/speedup: False
    dataset: Optional[str] = None
    split: Optional[str] = None
    seqlen: Optional[int] = None
    few_shot: int = 0
    extra_args: Optional[str] = None

    @field_validator("few_shot", mode="before")
    @classmethod
    def _few_shot_default(cls, v):
        return 0 if v is None else v


# --- structural pieces -----------------------------------------------------


class ExtensionPoint(BaseModel):
    """The cleanest place in the repo to express the idea."""
    file: str
    symbol: Optional[str] = None      # class / function to subclass / register / hook
    kind: ExtKind = "subclass"
    description: str = ""


class Ablation(BaseModel):
    """A switchable sub-component of the idea, for attribution."""
    name: str
    flag: str                         # the CLI/config flag or param that toggles it
    description: str = ""


class CorrectnessCheck(BaseModel):
    """A hard-gate check run before any eval (design §6, L0).

    `degenerate_params` are the settings under which the idea MUST reduce to the
    original algorithm (e.g. new compensation coefficient = 0 -> plain GPTQ); the
    strongest regression net. `closed_form_ref` names an independent closed form to
    cross-check the implementation against (never the repo itself)."""
    kind: CheckKind
    description: str = ""
    degenerate_params: dict = Field(default_factory=dict)
    closed_form_ref: Optional[str] = None


class LadderTier(BaseModel):
    """One rung of the cost ladder (design §6)."""
    name: TierName
    model: str                        # "random-2x1024" | "opt-125m" | "llama-7b" | ...
    protocol: EvalProtocol
    budget_gpu_hours: float = 0.0
    est_cost_usd: float = 0.0
    # Idea must clear this margin over baseline at this tier to advance. Defaults to
    # the claim's min_delta when unset.
    pass_margin: Optional[float] = None

    @property
    def is_expensive(self) -> bool:
        """Tiers that must be surfaced + approved before running (design §6)."""
        return self.name in ("L3_small", "L4_full") or self.budget_gpu_hours >= 1.0


class Baseline(BaseModel):
    """The repo's ORIGINAL algorithm — measured first, used as the grade reference
    and the eval-infra sanity probe (design §1.2, §5)."""
    method: str                       # e.g. "GPTQ"
    command: str                      # how to run the original
    # Optional sanity band: if the measured baseline falls outside this, blame the
    # eval infra, not the idea (BLOCKED). Absent -> no infra check available.
    sane_low: Optional[float] = None
    sane_high: Optional[float] = None


class Claim(BaseModel):
    """The single falsifiable claim graded per run (design §4)."""
    id: str
    statement: str                    # human-readable "idea beats baseline on X by Y"
    protocol: EvalProtocol
    min_delta: float                  # smallest improvement that counts as a real win
    tolerance: float = 0.0            # noise band; regress beyond it -> LOSE

    @field_validator("min_delta", "tolerance", mode="before")
    @classmethod
    def _nonneg(cls, v):
        v = 0.0 if v is None else float(v)
        return abs(v)


class IdeaSpec(BaseModel):
    """The machine-checkable spec produced by `ideaspec` and approved at gate 1."""
    idea_name: str
    target_repo: str                  # local path or github url (as given to ingest)
    summary: str = ""
    wins_on: WinsOn = "accuracy"
    extension_point: ExtensionPoint
    baseline: Baseline
    claim: Claim
    ablations: list[Ablation] = Field(default_factory=list)
    correctness: list[CorrectnessCheck] = Field(default_factory=list)
    ladder: list[LadderTier] = Field(default_factory=list)


# --- results ---------------------------------------------------------------


class Measurement(BaseModel):
    """A single measured number at a given tier for a given variant."""
    tier: TierName
    variant: str                      # "baseline" | "idea" | ablation name
    metric: str
    value: Optional[float] = None     # None -> the run failed to produce a number
    ok: bool = True
    log_path: Optional[str] = None


class Grade(BaseModel):
    """Pure-code go/no-go verdict (design §5)."""
    claim_id: str
    verdict: Verdict
    metric: str
    lower_is_better: bool
    baseline_value: Optional[float] = None
    idea_value: Optional[float] = None
    delta: Optional[float] = None     # signed improvement (positive = better)
    correctness_ok: bool = False
    infra_ok: bool = False
    deciding_tier: Optional[TierName] = None
    reason: str = ""
