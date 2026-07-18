# SpikeEval

*Spike a new-algorithm idea into any research repo and validate it against the repo's own
baseline — fast.*

Given a target research repo (an algorithm repo like GPTQ, or a framework like Quark) and a
new idea, SpikeEval implements the idea as a surgical diff, then fails fast and cheaply to
tell you — honestly — whether it actually beats the existing baseline.

Design doc: [docs/design/2026-07-18-spike-eval-agent-design.md](docs/design/2026-07-18-spike-eval-agent-design.md)

> Status: **scope A** — design + skeleton + interfaces + offline-testable pipeline. The
> agentic executors (idea→spec, surgical implementation, GPU baseline/validate/ablate) are
> interface stubs, filled in scope B.

## How it works

A deterministic pipeline, one run directory per idea, with two user gates plus an internal
correctness hard-gate:

```
ingest      normalize (repo, idea): clone/copy repo, capture idea (text | file | arxiv)
ideaspec    idea + repo -> falsifiable IdeaSpec (claim, extension point, ablations)  [GATE 1]
plan        feasibility + cost-ladder plan + SURFACE SCALE                            [GATE 2]
baseline    run the repo's ORIGINAL algorithm -> measured baseline (+ infra sanity)
implement   surgical diff at the extension point (isolated copy)
correctness HARD GATE: unit / closed-form / degenerate-equivalence (before any eval)
validate    cost ladder: L1 proxy -> [surface] -> L2 tiny -> L3 small
ablate      baseline / +idea / per-subcomponent -> attribution
grade       pure code: WIN / NEUTRAL / LOSE / BLOCKED (go / no-go)
report      analysis.md + bilingual README ; infra checklist ; failure attribution
```

Core principles baked in: **baseline-first**, **surface scale before spending**, **no
default `--yes`**, **correctness is a hard gate**, **diagnose eval-infra before blaming the
idea**, **pure-code judge isolated from execution**.

## Usage

```
spike-eval run <repo> <idea>     # repo: path|github url ; idea: text|file|arxiv id
spike-eval resume <run_dir>      # continue from the last gate / interruption
spike-eval report <run_dir>      # re-render the report from persisted state
```

## Development

```
uv sync
uv run pytest
uv run ruff check
```

Sibling project: [paper-reprise](../paper-reprise) — same deterministic-pipeline shape;
conventions were copied here (not imported) so the two evolve independently.
