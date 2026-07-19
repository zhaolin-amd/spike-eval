"""Run one claim: load Qwen3-8B, apply the MXFP4 fake-quant, evaluate with lm-eval.

Supports two claim families (detected by claim_id suffix):
  - Hellaswag accuracy  (claim_id ends in -hellaswag): prints acc_norm / acc
  - Wikitext PPL        (claim_id ends in -ppl):       prints word_perplexity

Only the final metric lines go to real stdout; all noisy model/harness logging
is redirected to stderr so the run's stdout.log is cleanly parseable.
"""
import os
import sys

# --- keep stdout clean: send everything to stderr until we print the metric ---
_real_stdout_fd = os.dup(1)
os.dup2(2, 1)


def _emit(line: str) -> None:
    os.write(_real_stdout_fd, (line + "\n").encode())


def method_for(claim_id: str):
    c = claim_id.lower()
    if "bf16" in c:
        return None  # baseline, no quantization
    if "mxfp4-quark-mbs-h-64" in c:
        return "MXFP4-Quark-MBS-H-64"
    if "mxfp4-quark-mbs-h-16bit" in c:
        return "MXFP4-Quark-MBS-H-16bit"
    if "mxfp4-quark-mbs-h-8bit" in c:
        return "MXFP4-Quark-MBS-H-8bit"
    if "mxfp4-quark-mbs-h" in c:
        return "MXFP4-Quark-MBS-H"
    if "mxfp4-quark-oas" in c:
        return "MXFP4-Quark-OAS"
    if "mxfp4-quark" in c:
        return "MXFP4-Quark"
    if "mxfp4-ocp" in c:
        return "MXFP4-OCP"
    if "mxfp4-16-oas" in c:   # check OAS before bare "16" to avoid prefix collision
        return "MXFP4-16-OAS"
    if "mxfp4-16" in c:
        return "MXFP4-16"
    if "mxfp4-mbs-s" in c:
        return "MXFP4-MBS-S"
    if "mxfp4-mbs-h-actw" in c:   # check ACTW before bare mbs-h (prefix collision)
        return "MXFP4-MBS-H-ACTW"
    if "mxfp4-mbs-h" in c:
        return "MXFP4-MBS-H"
    raise SystemExit(f"unknown claim id / method: {claim_id}")


def is_ppl_claim(claim_id: str) -> bool:
    return claim_id.endswith("-ppl")


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: eval_one.py <claim_id> [--smoke]")
    claim_id = sys.argv[1]
    smoke = "--smoke" in sys.argv[2:]
    method = method_for(claim_id)

    model_path = os.environ.get("PAPER_REPRISE_MODEL")
    if not model_path:
        raise SystemExit("PAPER_REPRISE_MODEL not set")
    tasks = os.environ.get("PAPER_REPRISE_TASKS", "hellaswag").split(",")
    tasks = [t.strip() for t in tasks if t.strip()]

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    _emit(f"Torch : {torch.__version__}")
    _emit(f"Transformers : {transformers.__version__}")
    _emit(f"lm_eval=={getattr(lm_eval, '__version__', 'unknown')}")
    print(f"[eval_one] claim={claim_id} method={method} tasks={tasks} smoke={smoke}",
          file=sys.stderr, flush=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True,
        )
    except TypeError:  # older transformers: dtype was torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
    model = model.to(device).eval()

    if method is not None:
        from qmodel import quantize_model_
        act_amax_map = None
        if method == "MXFP4-MBS-H-ACTW":
            # Activation-aware idea: load per-layer input-channel amax collected by
            # collect_act_amax.py; the dynamic weight search is weighted by amax^2.
            aa_path = os.environ.get("ACT_AMAX_PATH", "act_amax.pt")
            if not os.path.exists(aa_path):
                raise SystemExit(f"ACT_AMAX_PATH not found: {aa_path} "
                                 "(run collect_act_amax.py first)")
            act_amax_map = torch.load(aa_path, map_location="cpu")
            print(f"[eval_one] loaded act_amax for {len(act_amax_map)} layers from {aa_path}",
                  file=sys.stderr, flush=True)
        n = quantize_model_(model, method, act_amax_map=act_amax_map)
        print(f"[eval_one] quantized {n} linear layers with {method}",
              file=sys.stderr, flush=True)

    if is_ppl_claim(claim_id):
        # ── Wikitext word-level perplexity (Appendix tab:ppl_eval) ──────────────────
        # lm-eval's `wikitext` task reports word_perplexity and byte_perplexity;
        # the paper uses word-level PPL. bs=1 avoids padding artefacts on PPL.
        ppl_task = "wikitext"
        bs = 1
        limit = 4 if smoke else None
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs)
        results = lm_eval.simple_evaluate(
            model=lm, tasks=[ppl_task], num_fewshot=0, limit=limit, bootstrap_iters=0,
        )
        row = results["results"][ppl_task]
        wppl = None
        for k in ("word_perplexity,none", "word_perplexity"):
            if k in row and row[k] is not None:
                wppl = float(row[k])
                break
        if wppl is None:
            raise SystemExit(f"word_perplexity not found in wikitext results: {row}")
        _emit(f"word_perplexity: {wppl:.4f}")
        print(f"[eval_one] {ppl_task}: word_perplexity={wppl:.4f}", file=sys.stderr, flush=True)
    else:
        # ── Hellaswag / downstream accuracy ─────────────────────────────────────────
        bs = 1 if smoke else int(os.environ.get("EVAL_BATCH_SIZE", "32"))
        limit = 8 if smoke else None
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs)
        results = lm_eval.simple_evaluate(
            model=lm, tasks=tasks, num_fewshot=0, limit=limit, bootstrap_iters=0,
        )
        res = results["results"]
        primary = tasks[0]
        row = res[primary]

        def pct(key):
            for k in (f"{key},none", key):
                if k in row and row[k] is not None:
                    return float(row[k]) * 100.0
            return None

        accn = pct("acc_norm")
        acc = pct("acc")
        if accn is None and acc is None:
            raise SystemExit(f"no acc/acc_norm in results for {primary}: {row}")
        if accn is not None:
            _emit(f"acc_norm: {accn:.4f}")
        if acc is not None:
            _emit(f"acc: {acc:.4f}")
        print(f"[eval_one] {primary}: acc_norm={accn} acc={acc}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
