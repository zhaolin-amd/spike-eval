"""Collect per-input-channel activation amax for the SpikeEval activation-aware MBS idea.

Runs a tiny calibration set (pileval, num_calib_data=1 by default) through the *full-
precision* model and records, for every quantizable Linear, the per-input-channel
absolute-max of its input activation over all calibration tokens:

    a[k] = max over tokens t of |x[t, k]|      (shape (in_features,))

These amax vectors are what the idea squares (amax^2) and uses as the weighted-SSE
weights in the dynamic MBS weight-factor search (see mxquant._mbs_factor_dynamic).
Calibration-free direct-cast quant needs no such stats; introducing this one cheap
forward pass is the whole extra cost of the idea.

Output: torch.save({layer_name: amax_cpu_fp32}, out_path).

Env:
  PAPER_REPRISE_MODEL  base model path (Qwen3-8B snapshot)
  HF_HOME / HF_HUB_OFFLINE  point datasets at the local pileval cache
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qmodel import _quantizable


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def get_pileval_texts(num_calib_data: int):
    """First `num_calib_data` non-empty pileval validation texts (deterministic, no shuffle).

    Mirrors Quark's get_calib_dataloader_to_tensor pileval path
    (quark/torch/utils/llm/data_preparation.py): load_dataset(mit-han-lab/pile-val-backup,
    split=validation) then take text[:num_calib_data]."""
    from datasets import load_dataset

    ds = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    texts = []
    for t in ds["text"]:
        if t and t.strip():
            texts.append(t)
        if len(texts) >= num_calib_data:
            break
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-calib-data", type=int, default=1,
                    help="pileval samples (paper-simple validation uses 1).")
    ap.add_argument("--seqlen", type=int, default=512,
                    help="max tokens per sample (Quark pileval default 512).")
    ap.add_argument("--out", default="act_amax.pt")
    args = ap.parse_args()

    model_path = os.environ.get("PAPER_REPRISE_MODEL")
    if not model_path:
        raise SystemExit("PAPER_REPRISE_MODEL not set")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _log(f"[calib] loading tokenizer/model from {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to(device).eval()

    # Register forward-pre-hooks on exactly the layers quantize_model_ will quantize, so
    # the amax names line up 1:1 with the QuantLinear replacements.
    amax: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            x = inputs[0]
            k = x.shape[-1]
            a = x.detach().abs().reshape(-1, k).amax(dim=0).to(torch.float32)
            prev = amax.get(name)
            amax[name] = a if prev is None else torch.maximum(prev, a.to(prev.device))
        return hook

    targets = [(n, m) for n, m in model.named_modules() if _quantizable(n, m)]
    for name, mod in targets:
        handles.append(mod.register_forward_pre_hook(make_hook(name)))
    _log(f"[calib] hooked {len(targets)} quantizable linears")

    texts = get_pileval_texts(args.num_calib_data)
    _log(f"[calib] running {len(texts)} pileval sample(s), seqlen<={args.seqlen}")
    with torch.no_grad():
        for i, text in enumerate(texts):
            enc = tok(text, return_tensors="pt", truncation=True, max_length=args.seqlen)
            input_ids = enc.input_ids.to(device)
            model(input_ids)
            _log(f"[calib]   sample {i}: {input_ids.shape[1]} tokens")

    for h in handles:
        h.remove()

    out = {n: a.cpu() for n, a in amax.items()}
    torch.save(out, args.out)
    # Small sanity summary to stderr.
    any_name = next(iter(out))
    _log(f"[calib] saved {len(out)} layers -> {args.out}")
    _log(f"[calib] example {any_name}: shape={tuple(out[any_name].shape)} "
         f"amax_min={out[any_name].min():.4g} amax_max={out[any_name].max():.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
