"""ingest — normalize (repo, idea) into a run dir (design §2.1).

Two pure-ish helpers detect input kind; side effects (clone/copy the repo, fetch an
arxiv source) are injected so the pipeline stays testable offline.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

IdeaKind = Literal["text", "file", "arxiv"]
RepoKind = Literal["local", "github"]

_ARXIV_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_GITHUB_RE = re.compile(r"^(?:https?://)?(?:www\.)?github\.com/[\w.-]+/[\w.-]+", re.I)


@dataclass
class IngestInfo:
    repo_source: str
    repo_kind: RepoKind
    idea_kind: IdeaKind
    idea_source: str          # the raw arg (path / arxiv id / the text itself)
    arxiv_id: Optional[str] = None


def detect_idea_kind(idea_arg: str) -> tuple[IdeaKind, Optional[str]]:
    """Classify the idea argument. arxiv id/url -> ('arxiv', id); an existing file path
    -> ('file', None); anything else -> ('text', None)."""
    m = _ARXIV_RE.search(idea_arg.strip())
    if m and (idea_arg.strip().lower().startswith(("arxiv", "http")) or
              re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", idea_arg.strip())):
        return "arxiv", m.group(1)
    if Path(idea_arg).expanduser().is_file():
        return "file", None
    return "text", None


def detect_repo_kind(repo_arg: str) -> RepoKind:
    """github url -> 'github' ; otherwise treated as a local path."""
    return "github" if _GITHUB_RE.match(repo_arg.strip()) else "local"


def classify(repo_arg: str, idea_arg: str) -> IngestInfo:
    """Pure classification of the two inputs — no side effects."""
    idea_kind, arxiv_id = detect_idea_kind(idea_arg)
    return IngestInfo(
        repo_source=repo_arg,
        repo_kind=detect_repo_kind(repo_arg),
        idea_kind=idea_kind,
        idea_source=idea_arg,
        arxiv_id=arxiv_id,
    )


def default_fetch(info: IngestInfo, repo_dir: Path) -> None:
    """Default side-effecting acquisition: copy a local repo, or `git clone` a github
    url, into repo_dir. Injected into the pipeline so tests can pass a no-op."""
    if info.repo_kind == "local":
        src = Path(info.repo_source).expanduser().resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"local repo not found: {src}")
        # Copy contents into the (already-created) repo_dir, skipping heavy/VCS dirs.
        for child in src.iterdir():
            if child.name in (".git", "__pycache__", ".venv", "venv"):
                continue
            dest = repo_dir / child.name
            if child.is_dir():
                shutil.copytree(child, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(child, dest)
    else:
        subprocess.run(["git", "clone", "--depth", "1", info.repo_source, str(repo_dir)],
                       check=True)


def idea_slug_source(info: IngestInfo) -> str:
    """A human-ish string to slug the run dir from, known at ingest (before the spec):
    arxiv id, file stem, or the head of the free-text idea."""
    if info.idea_kind == "arxiv":
        return info.arxiv_id or "arxiv"
    if info.idea_kind == "file":
        return Path(info.idea_source).expanduser().stem
    return info.idea_source          # free text; RunDir.slug truncates + sanitizes


def read_idea_text(info: IngestInfo, fetch_arxiv: Optional[Callable] = None) -> str:
    """Resolve the idea to its raw text. 'text' -> itself ; 'file' -> file contents ;
    'arxiv' -> delegated to fetch_arxiv (injected; None -> a placeholder pointer)."""
    if info.idea_kind == "text":
        return info.idea_source
    if info.idea_kind == "file":
        return Path(info.idea_source).expanduser().read_text()
    if fetch_arxiv is not None:
        return fetch_arxiv(info.arxiv_id)
    return f"arxiv:{info.arxiv_id}\n\n(fetch not wired — scope B)"
