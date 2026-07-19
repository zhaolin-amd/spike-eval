"""GPTQ (IST-DASLab/gptq) OPT-family executors for SpikeEval.

The concrete injected callables for the bias-correction demo (design §8): patch the cloned
repo with the compensation term, run `opt.py`, and parse WikiText2 perplexity. Each unique
(model, alpha) run is cached under the run dir, so a full pipeline does only three real GPU
runs: pristine (baseline probe) / alpha=0 (degenerate check + ladder baseline) / alpha=X
(ladder idea).

Env knobs: SPIKE_EVAL_BC_ALPHA (idea strength, default 1.0), SPIKE_EVAL_GPTQ_WBITS (4),
SPIKE_EVAL_GPTQ_CALIB (wikitext2), SPIKE_EVAL_PYTHON (interpreter; default the current one,
which is verified to run opt.py on this node).
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from spike_eval import gpu, modelpaths
from spike_eval.correctness import CheckResult
from spike_eval.headless import call_claude
from spike_eval.implement import ImplementResult
from spike_eval.models import (
    Baseline, CorrectnessCheck, EvalProtocol, IdeaSpec, LadderTier, Measurement,
)
from spike_eval.pipeline import Executors
from spike_eval.rundir import RunDir

ASSET_PATCH = Path(__file__).resolve().parent.parent / "assets" / "gptq_bias_correction.patch"
GITHUB_URL = "https://github.com/IST-DASLab/gptq"

DEMO_ALPHA = float(os.environ.get("SPIKE_EVAL_BC_ALPHA", "1.0"))
WBITS = int(os.environ.get("SPIKE_EVAL_GPTQ_WBITS", "4"))
CALIB = os.environ.get("SPIKE_EVAL_GPTQ_CALIB", "wikitext2")
INTERP = os.environ.get("SPIKE_EVAL_PYTHON", sys.executable)
EVAL_DATASET = "wikitext2"
DEGENERATE_EPS = 1e-3          # alpha=0 must reproduce pristine ppl to within this
IMPLEMENT_TIMEOUT = float(os.environ.get("SPIKE_EVAL_IMPLEMENT_TIMEOUT", "1800"))

_DATASET_MARKERS = {"wikitext2", "ptb", "c4", "ptb-new", "c4-new"}


# --- stdout parsing (pure, offline-testable) -------------------------------


def parse_ppl(stdout: str, dataset: str = EVAL_DATASET) -> Optional[float]:
    """Extract a dataset's perplexity from opt.py stdout. Layout per section:
    `<dataset>` / `Evaluating ...` / integer layer indices / `<ppl float>`. The float
    BEFORE the marker (quantization time) is ignored by keying off the marker."""
    lines = stdout.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == dataset:
            for nxt in lines[idx + 1:]:
                s = nxt.strip()
                if s == "Evaluating ..." or s.isdigit() or not s:
                    continue
                if s in _DATASET_MARKERS:
                    return None
                try:
                    return float(s)
                except ValueError:
                    continue
    return None


def parse_layer_mse(stdout: str) -> Optional[float]:
    """Mean of the per-block `LAYER_MSE <i> <value>` lines the patched opt.py prints during
    quantization (the L1 proxy metric). None if no such line was seen."""
    vals: list[float] = []
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "LAYER_MSE":
            try:
                vals.append(float(parts[2]))
            except ValueError:
                continue
    return sum(vals) / len(vals) if vals else None


# --- run cache -------------------------------------------------------------


def _cache_key(model: str, alpha: Optional[float]) -> str:
    m = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[-24:]
    a = "pristine" if alpha is None else f"a{alpha}"
    return f"{m}_{a}"


def _run_and_cache(rd: RunDir, model: str, alpha: Optional[float], *, timeout: float = 1800
                   ) -> dict:
    """Run opt.py once for (model, alpha) with streaming early-termination after the
    WikiText2 ppl, parse BOTH the ppl (L2) and the mean layer MSE (L1) from the log, and
    cache them. One run therefore serves every tier. alpha=None -> pristine (no flag)."""
    key = _cache_key(model, alpha)
    cache = rd.ladder_dir / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    resolved = modelpaths.resolve_model(model)
    env_var, visible = gpu.visible_devices(1)
    if not visible:
        visible = "0"   # last-resort fallback; will fail loudly if no GPU present
    cmd = [INTERP, "opt.py", resolved, CALIB, "--wbits", str(WBITS)]
    if alpha is not None:
        cmd += ["--bc-alpha", str(alpha)]
    env = {**os.environ, env_var: visible, **modelpaths.hf_env_overlay()}

    log = rd.ladder_dir / f"{key}.log"
    ppl = _stream_run(cmd, rd.repo_dir, env, log, timeout=timeout)
    text = log.read_text() if log.exists() else ""
    data = {"model": model, "alpha": alpha, "ppl": ppl,
            "layer_mse": parse_layer_mse(text), env_var: visible}
    cache.write_text(json.dumps(data))
    return data


def _metric_value(data: dict, metric: str) -> Optional[float]:
    """Pick the tier's metric out of a cached run: 'perplexity' -> ppl, else layer MSE."""
    return data.get("ppl") if metric == "perplexity" else data.get("layer_mse")


def _stream_run(cmd: list[str], cwd: Path, env: dict, log: Path, *, timeout: float
                ) -> Optional[float]:
    """Stream opt.py stdout to `log`, capture the WikiText2 ppl, and kill the process
    (and its group) as soon as it appears — avoiding the later ptb/c4 eval (which needs
    dataset scripts unsupported by datasets>=4)."""
    seen_marker = False
    ppl: Optional[float] = None
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, text=True, bufsize=1,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            for line in proc.stdout:               # type: ignore[union-attr]
                lf.write(line)
                lf.flush()
                s = line.strip()
                if s == EVAL_DATASET:
                    seen_marker = True
                    continue
                if seen_marker:
                    if s == "Evaluating ..." or s.isdigit() or not s:
                        continue
                    try:
                        ppl = float(s)
                        break
                    except ValueError:
                        continue
        finally:
            _reap(proc)
    return ppl


def _reap(proc: subprocess.Popen) -> None:
    """Terminate the process group started with start_new_session and wait."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# --- injected executors ----------------------------------------------------


def baseline_executor(rd: RunDir, baseline: Baseline, protocol: EvalProtocol,
                      model: str) -> Measurement:
    """Infra-sanity probe: pristine (unpatched) GPTQ on the cheapest tier's model. Runs
    BEFORE implement patches the repo (pipeline order), so no --bc-alpha flag exists yet."""
    data = _run_and_cache(rd, model, alpha=None)
    ppl = data.get("ppl")
    return Measurement(tier="probe", variant="baseline", metric="perplexity",
                       value=ppl, ok=ppl is not None,
                       log_path=str(rd.ladder_dir / f"{_cache_key(model, None)}.log"))


def implementer(rd: RunDir, spec: IdeaSpec) -> ImplementResult:
    """Apply the hand-written bias-correction patch to the cloned repo (isolated copy)."""
    if not ASSET_PATCH.is_file():
        return ImplementResult(ok=False, notes=f"patch asset missing: {ASSET_PATCH}")
    patch_copy = rd.impl_dir / "idea.patch"
    patch_copy.write_text(ASSET_PATCH.read_text())
    proc = subprocess.run(["git", "apply", str(ASSET_PATCH)], cwd=str(rd.repo_dir),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return ImplementResult(ok=False, patch_path=str(patch_copy),
                               notes=f"git apply failed: {proc.stderr.strip()}")
    return ImplementResult(ok=True, patch_path=str(patch_copy),
                           files_touched=["opt.py"],
                           notes="bias-correction (--bc-alpha) grafted into opt_sequential")


def _implement_prompt(spec: IdeaSpec) -> str:
    """Prompt for the headless implementer. Pins the CLI contract the harness depends on
    (`--bc-alpha`, alpha=0 == vanilla GPTQ) while leaving the implementation autonomous."""
    ep = spec.extension_point
    return f"""You are implementing a new-algorithm idea directly in this cloned GitHub repo
(IST-DASLab/gptq, the original GPTQ code) at the current working directory. Make a small,
surgical edit — do not refactor unrelated code.

Idea to implement: {spec.summary}

Target file/location: `{ep.file}`, function `{ep.symbol}` ({ep.kind}).

You MUST honor this exact interface contract (a test and the eval harness depend on it):
1. Add a command-line argument `--bc-alpha` (type float, default 0.0) to {ep.file}.
2. When `--bc-alpha` is 0.0 (the default), behavior MUST be byte-for-byte identical to
   vanilla GPTQ: no extra work, no change to any weight or bias. This is checked by a
   degenerate-equivalence test (alpha=0 must reproduce the pristine WikiText2 perplexity
   exactly).
3. When `--bc-alpha` > 0: after GPTQ finishes quantizing the Linear layers of a decoder
   block (i.e. after the per-layer fasterquant calls in `{ep.symbol}`), apply bias
   correction — for each quantized Linear that has a bias, measure its mean output error on
   the calibration inputs (mean over calibration tokens of full-precision output minus
   quantized output, per output channel) and add `bc_alpha * error` to that Linear's bias.

Constraints:
- Edit ONLY `{ep.file}`. Do not touch datasets, eval, or other files.
- Keep the code runnable as-is (`python {ep.file} <model> wikitext2 --wbits 4`).
- Do not add prints other than what already exists (plus optional short debug is fine).
- Do not create new files. When done, stop.
"""


def headless_implementer(rd: RunDir, spec: IdeaSpec) -> ImplementResult:
    """Let headless Claude (`claude -p`) implement the idea as a surgical edit to the cloned
    repo, then verify a real diff was produced. The correctness hard-gate (alpha=0 ==
    pristine, bit-exact) is the safety net if the autonomous edit is wrong."""
    prompt = _implement_prompt(spec)
    (rd.impl_dir / "prompt.txt").write_text(prompt)
    code = call_claude(prompt, allowed_tools=["Read", "Edit", "Grep", "Bash"],
                       cwd=rd.repo_dir, timeout=IMPLEMENT_TIMEOUT)
    diff = subprocess.run(["git", "-C", str(rd.repo_dir), "diff"],
                          capture_output=True, text=True).stdout
    if not diff.strip():
        return ImplementResult(ok=False,
                               notes=f"headless produced no diff (claude exit={code})")
    patch = rd.impl_dir / "idea.patch"
    patch.write_text(diff)
    files = subprocess.run(["git", "-C", str(rd.repo_dir), "diff", "--name-only"],
                           capture_output=True, text=True).stdout.split()
    return ImplementResult(ok=True, patch_path=str(patch), files_touched=files,
                           notes=f"headless Claude implemented the idea (exit={code})")


def check_runner(rd: RunDir, check: CorrectnessCheck) -> CheckResult:
    """Degenerate-equivalence: patched repo with --bc-alpha 0 must reproduce the pristine
    WikiText2 ppl to within DEGENERATE_EPS. Other check kinds are not implemented here."""
    if check.kind != "degenerate_equivalence":
        return CheckResult(kind=check.kind, passed=False,
                           detail="unsupported check kind for gptq_opt")
    spec = rd.read_spec()
    model = spec.ladder[0].model if (spec and spec.ladder) else "facebook/opt-125m"
    pristine = _run_and_cache(rd, model, alpha=None).get("ppl")
    deg = _run_and_cache(rd, model, alpha=0.0).get("ppl")
    if pristine is None or deg is None:
        return CheckResult(kind=check.kind, passed=False, detail="a run produced no ppl")
    ok = abs(pristine - deg) <= DEGENERATE_EPS
    return CheckResult(kind=check.kind, passed=ok,
                       detail=f"pristine={pristine:.6f} alpha0={deg:.6f} "
                              f"|Δ|={abs(pristine - deg):.2e} eps={DEGENERATE_EPS:g}")


def tier_executor(rd: RunDir, tier: LadderTier, variant: str) -> Measurement:
    """Ladder measurement: baseline variant -> alpha 0 (== pristine by the degenerate
    gate); idea variant -> alpha DEMO_ALPHA. The tier's own metric selects L1 layer-MSE vs
    L2 perplexity from the single cached run."""
    alpha = 0.0 if variant == "baseline" else DEMO_ALPHA
    data = _run_and_cache(rd, tier.model, alpha=alpha)
    metric = tier.protocol.metric
    value = _metric_value(data, metric)
    return Measurement(tier=tier.name, variant=variant, metric=metric,
                       value=value, ok=value is not None,
                       log_path=str(rd.ladder_dir / f"{_cache_key(tier.model, alpha)}.log"))


def make_executors(implement: str = "patch") -> Executors:
    """Executors wired for the gptq-opt family. `fetch` uses the default git clone; the
    idea-spec is expected to already be on disk (hand-authored for the demo), so no
    spec_extractor is wired. `implement`: 'patch' applies the hand-written diff (derisked
    default); 'headless' lets `claude -p` write the diff autonomously."""
    if implement not in ("patch", "headless"):
        raise ValueError(f"implement must be 'patch' or 'headless', got {implement!r}")
    impl = headless_implementer if implement == "headless" else implementer
    return Executors(
        baseline=baseline_executor,
        implementer=impl,
        check_runner=check_runner,
        tier=tier_executor,
        ablation=None,
    )
