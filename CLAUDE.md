# CLAUDE.md — working principles for SpikeEval

Guidance for any agent (or human) working in this repo. These are distilled from working
rules established while building the sibling tool **paper-reprise** and the **quark**
feature work; they shaped SpikeEval's design and must keep holding as it grows. Each rule
lists *why* and *how it is (or should be) embodied here*.

SpikeEval spikes a new-algorithm idea into a research repo and validates it against the
repo's own baseline, fast — the value is an **honest go/no-go**, so these rules mostly
protect honesty and the user's control over cost and choice.

---

## Autonomy & user control

### 1. No default `--yes` — leave idea/model/config selection to the user
Do not auto-approve the pipeline's gates. Which idea variant, which model, which quant
config to validate is the user's call.
- **Why:** approval in one context does not extend to the next; silently picking hides the
  real unit of choice.
- **How here:** `cli._approve_spec` / `_approve_plan` default to `False`; the pipeline has
  gate 1 (idea-spec) and gate 2 (scale). Reserve auto-approval for a run the user has
  explicitly authorized (e.g. the committed demo driver, whose scale was surfaced first).

### 2. Surface scale BEFORE any large/expensive run — even under a standing "just run it"
"Don't pester me about which to pick" ≠ "launch a 70B × many-config job silently." Compute
cost is a separate concern from the selection gate.
- **Why:** a user who said "default is fine" still deserves a beat to stop or narrow a big
  job. (Origin: an AutoRound run fired all 20 claims × 11 tasks with no heads-up.)
- **How here:** `planstage.build_plan` / `format_scale` print per-tier model, config count,
  est. GPU-hours and cost; `validate.run_ladder` runs cheap tiers first and surfaces the
  scale **lazily at the first expensive tier** (gate 2). Even when approval is standing,
  *report* the scale (biggest model, tier count, rough cost) before spending. When unsure,
  default to a **subset** (smallest model / one representative config), not the full matrix.

### 3. When presenting a selection, show the full model × config — not just the model name
A claim is `model × quant-config × eval-protocol`. Show bit-width, group size, calibration
set, eval tasks/dataset, few-shot, target ± tolerance, and hardware.
- **Why:** the config is what makes the choice meaningful; "Llama-8B" alone hides it.
- **How here:** whenever you present options (AskUserQuestion or text), include the
  per-option quant config, calibration, eval protocol and target — mirror the IdeaSpec's
  `claim.protocol` + ladder tiers, not just a name.

### 4. Validate the cheapest core-mechanism target first — not the heavy headline claim
When deciding *what* to validate, aim the claim at the cheapest signal that can falsify the
idea (a proxy metric / small model / the core mechanism), not the heaviest headline number;
then surface the chosen scope (model × config × cost) to the user.
- **Why:** an idea/paper usually points at a heavy headline metric, but a cheap core-
  validation kills a wrong idea far sooner and cheaper. (Origin: TurboQuant specextract
  emitted the heavy KV-cache LongBench claims; the faithful cheap target was §4.1 distortion
  rates.)
- **How here:** the cost ladder puts the cheapest falsifying tier first (L1 proxy / tiny
  model); author the `claim` around the core mechanism, and surface the scope at gate 1/2
  rather than defaulting to the biggest number.

---

## Quality gates (never skip on "looks fine")

### 5. Self-review / correctness is a HARD gate — green tests + green lint do not substitute
Tests passing and lint clean is necessary, not sufficient. Run the explicit review/check
step before committing.
- **Why:** "looks fine" is exactly when real issues slip through. (Origin: an
  NVFP4→MXFP4 feature committed on green tests+lint; a later self-review found 3 real bugs
  + ruff errors, all needing rework.)
- **How here:** the `correctness` stage is a non-skippable gate before any eval; an idea
  that declares **zero** correctness checks does NOT clear it (→ BLOCKED). Correct order:
  tests green → lint green → self-review → fix findings → commit → push.

### 6. Cross-check the algorithm against an INDEPENDENT closed form, in unit tests
Do not only check the measured metric against a target — that is circular. Find a
textbook/closed-form result the algorithm must also satisfy and assert it.
- **Why:** a wrong impl can "match" a number by luck but cannot satisfy an independent
  identity. (Origin: TurboQuant's MSE distortion *is* the Lloyd–Max Gaussian MSE.)
- **How here:** prefer a **degenerate-equivalence** check (a knob value that must reduce
  the idea exactly to the baseline — e.g. `--bc-alpha 0` == vanilla GPTQ, verified
  bit-exact) and/or a `closed_form_ref`. Validate the independent result first, the target
  second.

### 7. The source of truth is the idea's paper/spec — not an upstream/reference repo
When an idea comes from a paper (arxiv input) that builds on a prior method with its own
repo, implement the method as the **paper/spec** defines it, even if an upstream/reference
repo uses a different convention. Reference repos are read-only aids, never the authority,
and never a back door to the target number.
- **Why:** an upstream repo can diverge from the paper's stated definition; the paper is
  what's being validated. (Origin: TurboQuant builds on QJL — implement the paper's restated
  definition, not QJL's convention.)
- **How here:** the arxiv ingest path takes only the *method*; `idea_spec.public.yaml` is a
  redacted view so the implementer can't read the target band; keep any upstream repo as a
  read-only reference alongside the spec.

### 8. When a number is off, diagnose the eval infrastructure BEFORE blaming the idea
A moved metric is only trustworthy once the baseline itself is sane.
- **Why:** the gap is often the harness, not the algorithm. (Origin: MXFP4 acc_norm was
  off because a pre-instantiated HF model was passed to lm-eval instead of a path; the
  BF16 baseline itself was off by 1.55, which ruled out the quantization. PPL was
  insensitive; log-likelihood ranking was not.)
- **How here:** `baseline` is measured first as an **infra-sanity probe**
  (`baseline.infra_ok` + the spec's `sane_low/high`); a baseline out of band → BLOCKED,
  suspect the protocol. Baseline and idea run the **same code path / same eval**, differing
  only by the idea's flag, so the comparison is fair.

---

## Reporting

### 9. Order a toolchain hardware→software / produce→evaluate
List environment components top-down along the stack: **CUDA/ROCm → torch → transformers
→ lm_eval** (accelerator runtime → framework → library → eval harness).
- **Why:** it mirrors how the stack is built and how a run flows (quantize → evaluate), so
  it reads naturally — not alphabetical or arbitrary.
- **How here:** any Environment line in a report (`report.py`) uses this order.

### 10. Write analysis in separate `analysis_en.md` / `analysis_zh.md`, one language each
English gap analysis in `analysis_en.md`, Chinese in `analysis_zh.md`; each is embedded
into `README.md` / `README_zh.md` respectively. Do NOT create one bilingual `analysis.md`.
- **Why:** the report renderer pulls the matching language; a mixed file duplicates and
  drifts. Keep analysis concise and non-redundant with the auto-generated verdict.
- **Structure:** root-cause (a table comparing baseline vs idea setup) → evidence → why
  other metrics are unaffected → fix direction.
- **How here:** `report.render_reports` emits `analysis_en.md` + `analysis_zh.md` (plus
  `README.md` / `README_zh.md`); keep them one-language-per-file when extending reports.

### 11. Add an algorithm-overview diagram to the report
For a non-trivial idea, add a diagram so a reader grasps *how* the method works without
reading the code (the results table shows *what*, the diagram shows *how/why*).
- **How:** generate with a **committed** script under `figures/` (e.g.
  `figures/gen_<name>_diagram.py`), save `figures/<name>.png`, embed in both READMEs.
  Use **matplotlib** (reliable on these nodes; HTML+SVG screenshotting fails — missing
  `libatk`). `FancyBboxPatch` for rounded blocks, `annotate('', arrowprops=...)` for
  arrows, `rcParams['mathtext.fontset']='cm'` and `$...$` mathtext for formulas (never
  Unicode subscript glyphs). Clear gaps between boxes; arrow endpoints outside boxes; never
  route an arrow through a box (reserve a side lane for loop-backs). matplotlib usually is
  not in the run venv — use `/home/zhaolin/miniconda3/envs/quark_cuda/bin/python`.
  **Always Read the rendered PNG back** and fix glyphs/overlaps/arrows before committing.

---

## Git workflow

### 12. Pull before you push
**Always `git pull` (fast-forward, or `--rebase` if you have local commits) before
`git push`** — never push straight from a local checkout that hasn't just synced with
`origin/main`.
- **Why:** caught in practice on a sibling repo (paper-reprise) — a session committed and
  nearly pushed on a base 24 commits behind `origin/main`. Multiple sessions/agents can work
  in these repos, so a checkout goes stale between when you started and when you're ready
  to push.
- **How here:** before any `git push`, run `git fetch` + check `git log HEAD..origin/main`;
  if behind, pull first (fast-forward when there are no local commits yet, `--rebase` when
  there are). Applies mid-session too, not just at session start.

---

## Environment (this node)

- Pretrained models: read-first from `/group/amdneuralopt/huggingface/pretrained_models`
  (`<org>/<model>` snapshot layout); downloads go to `/scratch/$USER/pretrained_models`
  (never `$HOME`, small quota). Encoded in `modelpaths.py` (env `SPIKE_EVAL_MODEL_BASE` /
  `SPIKE_EVAL_DOWNLOAD_DIR`).
- H200 × 8 node; pick free GPUs via `gpu.py`. The GPTQ repo runs under the
  `quark_cuda` conda interpreter (torch 2.9 + transformers 4.57); point
  `SPIKE_EVAL_PYTHON` at it while driving the pipeline from the project venv.
