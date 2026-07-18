"""report — analysis_en.md / analysis_zh.md + bilingual README (design §2.1, CLAUDE.md §8).

Always renders the MEASURED delta, never a hoped-for one. Includes the infra-sanity
verdict and failure attribution so a null result is as informative as a win. Analysis is
split one-language-per-file (English in analysis_en.md, Chinese in analysis_zh.md).
"""
from __future__ import annotations

from spike_eval.models import Grade, IdeaSpec

_VERDICT_EMOJI = {"WIN": "✅", "NEUTRAL": "➖", "LOSE": "❌", "BLOCKED": "⛔"}


def _delta_str(g: Grade) -> str:
    if g.delta is None:
        return "n/a"
    direction = "lower better" if g.lower_is_better else "higher better"
    return f"{g.delta:+.6g} ({direction})"


def render_analysis(spec: IdeaSpec, grade: Grade, *, zh: bool = False) -> str:
    """Structured analysis, one language per file (design §5, §7; CLAUDE.md §8)."""
    e = _VERDICT_EMOJI.get(grade.verdict, "")
    ep = (f"`{spec.extension_point.file}` :: `{spec.extension_point.symbol}` "
          f"({spec.extension_point.kind})")
    if zh:
        lines = [
            f"# 分析 — {spec.idea_name}",
            "",
            f"**结论:{e} {grade.verdict}** — {grade.reason}",
            "",
            "## 实测",
            f"- 指标:`{grade.metric}`（{'越低越好' if grade.lower_is_better else '越高越好'}）",
            f"- baseline:`{grade.baseline_value}`",
            f"- idea:`{grade.idea_value}`",
            f"- delta:`{_delta_str(grade)}`",
            f"- 判定台阶:`{grade.deciding_tier}`",
            "",
            "## Gate",
            f"- 正确性 gate:{'通过' if grade.correctness_ok else '未通过'}",
            f"- eval-infra sanity:{'通过' if grade.infra_ok else '未通过'}",
            "",
            "## 命题",
            f"- {spec.claim.statement}",
            f"- min_delta（真实 win 阈值):`{spec.claim.min_delta}`;tolerance（噪声带):"
            f"`{spec.claim.tolerance}`",
            "",
            "## Extension point",
            f"- {ep}",
        ]
        return "\n".join(lines) + "\n"
    lines = [
        f"# Analysis — {spec.idea_name}",
        "",
        f"**Verdict: {e} {grade.verdict}** — {grade.reason}",
        "",
        "## Measured",
        f"- metric: `{grade.metric}` ({'lower' if grade.lower_is_better else 'higher'} is better)",
        f"- baseline: `{grade.baseline_value}`",
        f"- idea: `{grade.idea_value}`",
        f"- delta: `{_delta_str(grade)}`",
        f"- deciding tier: `{grade.deciding_tier}`",
        "",
        "## Gates",
        f"- correctness gate: {'PASS' if grade.correctness_ok else 'FAIL'}",
        f"- eval-infra sanity: {'PASS' if grade.infra_ok else 'FAIL'}",
        "",
        "## Claim",
        f"- {spec.claim.statement}",
        f"- min_delta (real win): `{spec.claim.min_delta}` ; tolerance (noise): "
        f"`{spec.claim.tolerance}`",
        "",
        "## Extension point",
        f"- {ep}",
    ]
    return "\n".join(lines) + "\n"


def render_readme(spec: IdeaSpec, grade: Grade, *, zh: bool) -> str:
    e = _VERDICT_EMOJI.get(grade.verdict, "")
    if zh:
        head = [
            f"# {spec.idea_name}",
            "",
            f"在 `{spec.target_repo}` 上验证新算法 idea 是否胜过原 baseline。",
            "",
            f"## 结论:{e} {grade.verdict}",
            f"{grade.reason}",
            "",
            "| 项 | 值 |",
            "|---|---|",
            f"| 指标 | `{grade.metric}`（{'越低越好' if grade.lower_is_better else '越高越好'}）|",
            f"| baseline | `{grade.baseline_value}` |",
            f"| idea | `{grade.idea_value}` |",
            f"| delta | `{_delta_str(grade)}` |",
            f"| 判定台阶 | `{grade.deciding_tier}` |",
            f"| 正确性 gate | {'通过' if grade.correctness_ok else '未通过'} |",
            f"| eval-infra sanity | {'通过' if grade.infra_ok else '未通过'} |",
        ]
        return "\n".join(head) + "\n"
    head = [
        f"# {spec.idea_name}",
        "",
        f"Validation of a new-algorithm idea against the baseline in `{spec.target_repo}`.",
        "",
        f"## Verdict: {e} {grade.verdict}",
        f"{grade.reason}",
        "",
        "| item | value |",
        "|---|---|",
        f"| metric | `{grade.metric}` ({'lower' if grade.lower_is_better else 'higher'} better) |",
        f"| baseline | `{grade.baseline_value}` |",
        f"| idea | `{grade.idea_value}` |",
        f"| delta | `{_delta_str(grade)}` |",
        f"| deciding tier | `{grade.deciding_tier}` |",
        f"| correctness gate | {'PASS' if grade.correctness_ok else 'FAIL'} |",
        f"| eval-infra sanity | {'PASS' if grade.infra_ok else 'FAIL'} |",
    ]
    return "\n".join(head) + "\n"


def render_reports(spec: IdeaSpec, grade: Grade) -> tuple[str, str, str, str]:
    """Return (analysis_en, analysis_zh, readme_en, readme_zh)."""
    return (render_analysis(spec, grade, zh=False),
            render_analysis(spec, grade, zh=True),
            render_readme(spec, grade, zh=False),
            render_readme(spec, grade, zh=True))
