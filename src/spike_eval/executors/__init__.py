"""Repo-family executor implementations — the concrete injected callables that make the
deterministic pipeline actually clone, patch, and run a target repo.

`gptq_opt` is the first family (IST-DASLab/gptq, OPT models), driving the bias-correction
demo (design §8).
"""
