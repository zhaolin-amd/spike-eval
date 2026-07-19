# SpikeEval — Algorithm-Idea Spike & Validation Agent — Design Doc

> Date: 2026-07-18
> Status: design approved (scope A: design + skeleton); MVP (scope B) pending
> Sibling: [paper-reprise](../../../paper-reprise) — same deterministic-pipeline shape,
> conventions copied (not imported). paper-reprise *reproduces a paper's number*;
> SpikeEval *validates whether a new idea beats the repo's own baseline*.

## 1. Goal and Scope

Build an agent that, given **a target research repo** (an algorithm repo like GPTQ, or a
framework like Quark) **plus a new-algorithm idea**, implements the idea as a surgical
change on that repo and **validates, fast, whether it actually beats the repo's existing
baseline** — reporting an honest go / no-go with the evidence.

- The idea can be handed in three ways, auto-detected by `ingest`: **free-text**, a
  **written idea-spec file** (md/yaml), or an **arxiv id/url** describing the improvement.
  An arxiv idea contributes only its *method*, grafted onto the target repo and judged
  against **that repo's measured baseline** — never the paper's reported number (that is
  paper-reprise's job); the boundary keeps the two tools distinct.
- The target repo can be a **local path** or a **GitHub url** (cloned into the run dir).

### 1.1 Core Insight

Fast validation is not "run the code". It is two things:

1. Turning a one-line idea into a **machine-checkable, falsifiable claim** — a metric, a
   direction, a `min_delta` that counts as a real win, measured **against the repo's own
   baseline** on a fixed protocol; and finding the **cleanest extension point** in the repo
   to express the idea as a small, reversible diff.
2. **Failing fast and cheaply, in the right order** — cheap correctness gates before any
   eval, a cost ladder (layer-proxy → tiny → small) that kills bad ideas before they cost
   real compute, and honest attribution when a number moves (idea? infra? noise?).

Everything else (PPL, lm-eval-harness, the repo's own eval scripts) is already
standardized in the quant field and is leaned on rather than rebuilt.

### 1.2 Locked Design Decisions

| Dimension | Decision |
|---|---|
| Idea input | Free-text **and** idea-spec file **and** arxiv — `ingest` auto-detects, normalizes to one `IdeaSpec` |
| Target repo | Local path **or** GitHub url; cloned/copied into the run dir (never mutated in place) |
| Baseline | **Baseline-first**: the repo's *original* algorithm is measured before the idea is implemented; the idea is graded against *that measured number*, not a paper's |
| Claim | One **falsifiable** claim per run: metric + direction + `min_delta` + fixed eval protocol |
| Implementation | headless Claude writes a **surgical diff** at a named extension point, in an isolated copy; degenerate-parameter equivalence to the original is the primary regression net |
| Correctness | A **hard gate before any eval**: unit / closed-form cross-check / degenerate-equivalence must pass, else BLOCKED (mirrors the self-review-is-a-hard-gate principle) |
| Validation | A **cost ladder** L1 layer-proxy → L2 tiny → L3 small; each tier has a pass band and a budget; advance only on pass |
| Scale | **Surface scale before any expensive tier** (tier model / configs / est. GPU-h / est. cost) and gate on it — never silently run a big model |
| Judge | **Pure-code, isolated from execution**: WIN requires correctness-passed AND infra-sane AND delta ≥ `min_delta` in the right direction |
| Autonomy | Semi-auto: gate 1 (idea-spec + claim approval) + gate 2 (plan / scale approval); selection left to the user, no default `--yes` |
| Infra vs algo | Report a moved number only after an **eval-infra sanity checklist** passes; a bad baseline blames infra, not the idea |
| Deployment | Manual CLI, per-idea, file-based state; one isolated run dir per idea |
| Report | Bilingual `README.md` (en) / `README_zh.md` (zh) + a structured `analysis.md`; always the measured delta, never a hoped-for one |

### 1.3 Architecture Choice

A **deterministic pipeline where the agent only enters the open-ended stages**
(idea-spec extraction, implementation, and — later — env setup). The hard constraints
(falsifiable pure-code judge, two-gate semi-autonomy, surface-scale-before-spend,
reproducible isolated runs) all point to a deterministic skeleton that confines model
nondeterminism to the genuinely open steps. Grading is pure-code, never handed to an LLM.

## 2. Overall Architecture

One CLI invocation = one idea = one run directory. No queue, no DB, no cron.

```
spike-eval run <repo> <idea>     # repo: path|github url ; idea: text|file|arxiv id
spike-eval resume <run_dir>      # resume from last interruption / gate
spike-eval report <run_dir>      # re-render the report from persisted state
```

### 2.1 Stages (map to `src/spike_eval/*.py`)

```
ingest        normalize (repo, idea) -> clone/copy repo/, write ingest.json + idea.md ;
              detect input kind ; slug the run dir from the idea (text head/stem/arxiv id)
ideaspec      headless Claude: idea + repo -> IdeaSpec (falsifiable claim, extension point,
              ablations, correctness checks, ladder tiers)          [GATE 1: approve spec]
plan          feasibility + COST LADDER plan ; record surfaced scale to plan.json
baseline      run the repo's ORIGINAL algorithm once -> infra-sanity probe on the cheapest
              tier's model (the ladder re-measures baseline per tier for the fair compare)
implement     headless Claude: surgical diff at extension point (isolated copy) -> impl patch
correctness   HARD GATE: unit / closed-form / degenerate-equivalence  (before any eval)
validate      cost ladder: L1 proxy -> L2 tiny -> [GATE 2 at first expensive tier] ->
              L3 small ; fail fast on the pass margin
ablate        baseline / +idea / per-subcomponent -> attribution
grade         pure code, isolated: WIN / NEUTRAL / LOSE / BLOCKED  (go / no-go)
report        analysis.md + bilingual README ; infra checklist ; failure attribution
```

Two user gates (idea-spec at gate 1; the scale at gate 2, surfaced lazily just before the
first expensive tier) plus the internal correctness hard-gate. `baseline`, `implement`,
`correctness`, `validate`, `ablate` side effects are **injected callables**, so the whole
pipeline is testable offline with fakes (as in paper-reprise).

## 3. Run directory layout

`runs/<idea-slug>-<repo-name>-<timestamp>/`

```
ingest.json            input metadata (repo source, idea kind)
idea.md                the raw idea text / fetched source            [tracked]
idea_spec.yaml         extracted IdeaSpec (single private spec)      [tracked]
plan.json              cost-ladder plan + surfaced scale             [tracked]
repo/                  cloned/copied target repo                     [gitignored]
env/                   dedicated venv                                [gitignored]
baseline/              baseline metrics + stdout                     [tracked json/log]
impl/                  the surgical diff (patch) + notes             [tracked]
correctness/           gate results (unit/closed-form/degenerate)    [tracked]
ladder/                per-tier validate outputs                     [tracked json/log]
ablation/              per-variant results                           [tracked]
grade.json             pure-code verdict                             [tracked]
analysis.md            structured analysis (gaps, attribution)       [tracked]
README.md / README_zh.md   bilingual report                         [tracked]
```

Model weights, cloned repo, venv, and any `*.safetensors/.bin/...` are gitignored; the
whole `runs/` tree of heavy folders is re-fetchable. Deleting a run dir removes everything
that run touched.

## 4. Key contracts (`models.py`)

- `ExtensionPoint` — file, symbol, kind (`subclass|register|hook|core-edit`); prefer
  low-blast-radius kinds so the baseline stays untouched and the diff is reversible.
- `Claim` — falsifiable: an `EvalProtocol` (which carries `metric`, `lower_is_better`,
  `dataset`, …), plus `min_delta` (the smallest change that counts as a real win) and
  `tolerance` (noise band). `min_delta` and `tolerance` are independent knobs, so the
  NEUTRAL dead-zone `[-tolerance, min_delta)` need not be symmetric.
- `CorrectnessCheck` — `kind` (`unit|closed_form|degenerate_equivalence`),
  `degenerate_params` (settings under which the idea must reduce to the original), and an
  optional independent `closed_form_ref` to cross-check against.
- `LadderTier` — `name` (`L1_proxy|L2_tiny|L3_small|L4_full`), `model`, an `EvalProtocol`,
  `budget_gpu_hours` + `est_cost_usd`, and `pass_margin` (defaults to the claim's
  `min_delta`); `is_expensive` (name in {L3,L4} or budget ≥ 1 GPU-h) flags tiers that must
  be surfaced before running.
- `Baseline` — how to run the repo's original algorithm; its measured number is the grade
  reference and the infra-sanity probe.
- `IdeaSpec` — the whole thing: idea name, target repo, hypothesis (wins on
  accuracy/bits/speed/…), extension point, baseline, claim, ablations, correctness checks,
  ladder tiers.
- `Grade` — `verdict` (`WIN|NEUTRAL|LOSE|BLOCKED`), measured baseline vs idea, delta,
  whether correctness and infra sanity passed, and a human reason.

## 5. Grade logic (pure code, `grade.py`)

A claim is graded **only** on measured numbers already on disk:

1. `BLOCKED` if correctness gate failed (incl. **zero checks declared**), infra sanity
   failed (baseline out of band → suspect the eval protocol, not the idea), no ladder tier
   was defined, or no tier produced a comparable measurement (e.g. the only tiers were
   expensive and declined at gate 2).
2. Else compare idea vs measured baseline in the claim's direction:
   - `WIN`   — improves by ≥ `min_delta`.
   - `LOSE`  — regresses beyond `tolerance`.
   - `NEUTRAL` — within the noise band (no evidence of a real win).

No LLM in the judge. Same shape as paper-reprise's process-faithful + value-in-tolerance
rule, adapted from "matches the paper" to "beats the baseline".

## 6. Cost ladder & surface-scale

Ladder tiers, cheapest first; each kills a class of bad idea before the next spends more:

| Tier | What runs | Kills |
|---|---|---|
| L0 correctness | unit / closed-form / degenerate-equivalence (pure gate, `correctness.py`) | implementation bugs |
| L1 proxy | one layer / small random tensors: reconstruction MSE, quant error | wrong-direction ideas |
| L2 tiny | 125M–1.3B: ppl on a small corpus | not-accurate-enough ideas |
| L3 small | 7B: ppl + 1–2 tasks | doesn't-scale ideas |
| L4 full | target model: full eval | (final headline number) |

`plan` computes the ladder and records the **scale** (per-tier model, est. GPU-hours, est.
cost) to `plan.json`. The ladder runs cheap tiers automatically and **surfaces the scale at
gate 2 only when it reaches the first `is_expensive` tier** — declining does NOT abort the
run; it stops with the cheap-tier evidence already gathered and reports the skipped tiers.
This encodes "surface scale before large runs", "no default `--yes`", and "cheap signals
first" together.

## 7. Autonomy & guardrails

- **Gate 1** — after `ideaspec`: user reviews/edits `idea_spec.yaml` (claim, extension
  point, ladder) and approves. Resuming from a run dir *is* the approval.
- **Gate 2** — surfaced lazily, just before the first expensive tier: user approves the
  scale. Declining stops the ladder and grades on the cheap-tier evidence (not an abort).
- **Correctness hard-gate** — internal, non-skippable; green tests+lint never substitute
  for the degenerate-equivalence / closed-form check. An idea that declares **no**
  correctness check does NOT clear the gate (→ BLOCKED); at least one is required to enter
  eval.
- Per-tier budgets & timeouts; isolated run dir; deterministic seeds where possible.

## 8. Scope B (next, not this turn)

- Fill the agentic executors: `ideaspec` (idea→spec headless), `implement` (surgical-diff
  headless in an isolated repo copy), `baseline` / `validate` / `ablate` GPU executors,
  reusing paper-reprise's setup-loop pattern (build env until the repo's own eval smoke-
  tests green).
- Drive one concrete end-to-end example (e.g. a compensation term added to GPTQ) through
  `ingest → … → report` to falsify the core risk before generalizing.
