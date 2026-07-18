"""Wrapper around `claude -p` headless invocation (copied from paper-reprise).

Success is determined by the expected output file appearing — NOT by exit code,
because the skill can exit 0 while silently failing to write.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class HeadlessResult:
    ok: bool
    output_path: Optional[Path] = None
    error: str = ""
    exit_code: int = 0


def _call_claude(prompt: str, allowed_tools: list[str], cwd: Path,
                 timeout: float | None = None) -> int:
    """Invoke `claude -p`, prompt via stdin. Returns exit code (124 on timeout)."""
    try:
        proc = subprocess.run(
            ["claude", "-p", "--permission-mode", "acceptEdits",
             "--allowedTools", ",".join(allowed_tools)],
            input=prompt, text=True, cwd=str(cwd), timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124
    return proc.returncode


def run_headless(prompt: str, allowed_tools: list[str], cwd: Path,
                 expect_file: Path, timeout: float | None = None) -> HeadlessResult:
    code = _call_claude(prompt, allowed_tools, cwd, timeout=timeout)
    if expect_file.exists():
        return HeadlessResult(ok=True, output_path=expect_file, exit_code=code)
    return HeadlessResult(ok=False, exit_code=code,
                          error=f"expected output {expect_file} did not appear "
                                f"(exit={code})")
