"""L1 cheap proxy: per-layer output-reconstruction error, baseline vs activation-aware idea.

Runs ONE calibration forward through the FP model. For each quantizable Linear, with the
real (activation-quantized) input Xq fixed, it compares the weight-quantization error that
propagates to the layer output:

    ref      = Xq @ W_fp^T                       (ideal output given this input)
    err_base = MSE(Xq @ Wq_base^T, ref)          (plain-SSE dynamic weight search)
    err_idea = MSE(Xq @ Wq_idea^T, ref)          (amax^2-weighted dynamic weight search)

Both variants share the identical activation quant (static MBS) and identical inner OAS
scale, so the only difference is the weight-factor search objective — this isolates the
idea. It is a *fair* proxy: the idea directly minimizes a diagonal surrogate
sum_k amax_k^2 (dW_k)^2, whereas this measures the true output error E[(dW x)^2] on real
activations (cross-channel terms and the real per-token distribution included), which the
idea does not directly optimize.

Reports aggregate NMSE (sum err / sum ref-energy) for both, the relative improvement, and
the fraction of layers where the idea wins. Cheap: no lm-eval, one forward pass.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

from mxquant import fake_quant
from qmodel import _quantizable


def _log(m):
    print(m, file=sys.stderr, flush=True)


def main() -> int:
    model_path = os.environ.get("PAPER_REPRISE_MODEL")
    if not model_path:
        raise SystemExit("PAPER_REPRISE_MODEL not set")
    aa_path = os.environ.get("ACT_AMAX_PATH", "act_amax.pt")
    act_amax_map = torch.load(aa_path, map_location="cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to(device).eval()

    # OAS static-activation / dynamic-weight config = the MBS-H family both variants share.
    ACT = dict(mbs="static", oas=True, oas_block=16, macro_block=128)
    WT = dict(mbs="dynamic", oas=True, oas_block=16, macro_block=128)

    stats = {"err_base": 0.0, "err_idea": 0.0, "ref": 0.0,
             "n_layers": 0, "n_idea_wins": 0}
    per_layer = []

    def make_hook(name: str):
        def hook(module, inputs):
            x = inputs[0]
            w_fp = module.weight.data.to(torch.float32)
            xq = fake_quant(x.to(torch.float32), **ACT)
            ref = F.linear(xq, w_fp)
            wq_base = fake_quant(w_fp, **WT)
            aa = act_amax_map.get(name)
            chan_w = (aa.to(torch.float32) ** 2) if aa is not None else None
            wq_idea = fake_quant(w_fp, chan_w=chan_w, **WT)
            e_base = ((F.linear(xq, wq_base) - ref) ** 2).sum().item()
            e_idea = ((F.linear(xq, wq_idea) - ref) ** 2).sum().item()
            r = (ref ** 2).sum().item()
            stats["err_base"] += e_base
            stats["err_idea"] += e_idea
            stats["ref"] += r
            stats["n_layers"] += 1
            if e_idea < e_base:
                stats["n_idea_wins"] += 1
            per_layer.append((name, e_base / (r + 1e-30), e_idea / (r + 1e-30)))
        return hook

    handles = [m.register_forward_pre_hook(make_hook(n))
               for n, m in model.named_modules() if _quantizable(n, m)]
    _log(f"[l1] hooked {len(handles)} layers")

    text = _first_pileval_text()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        model(enc.input_ids.to(device))
    for h in handles:
        h.remove()

    nmse_base = stats["err_base"] / stats["ref"]
    nmse_idea = stats["err_idea"] / stats["ref"]
    rel = (nmse_base - nmse_idea) / nmse_base * 100.0
    # Report on real stdout (parseable), detail on stderr.
    print(f"L1_nmse_base: {nmse_base:.6e}")
    print(f"L1_nmse_idea: {nmse_idea:.6e}")
    print(f"L1_rel_improvement_pct: {rel:+.3f}")
    print(f"L1_idea_wins: {stats['n_idea_wins']}/{stats['n_layers']}")
    per_layer.sort(key=lambda t: (t[1] - t[2]), reverse=True)
    _log("[l1] top-8 layers by idea improvement (name, nmse_base, nmse_idea):")
    for name, b, i in per_layer[:8]:
        _log(f"[l1]   {name:45s} {b:.4e} -> {i:.4e}")
    _log("[l1] worst-4 (idea hurts most):")
    for name, b, i in per_layer[-4:]:
        _log(f"[l1]   {name:45s} {b:.4e} -> {i:.4e}")
    return 0


def _first_pileval_text():
    from datasets import load_dataset
    ds = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    for t in ds["text"]:
        if t and t.strip():
            return t
    raise SystemExit("no pileval text")


if __name__ == "__main__":
    sys.exit(main())
