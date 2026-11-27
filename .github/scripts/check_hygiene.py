#!/usr/bin/env python3
"""Repository hygiene, privacy, size, and study.yaml manifest checks."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[2]
MAX_FILE_BYTES = 15 * 1024 * 1024  # 15 MB ceiling

FORBIDDEN_FILENAME_PATTERNS = [
    re.compile(r"^\.api_key$"),
    re.compile(r"^\.env(\..+)?$"),
    re.compile(r"^usa_\d+\.csv(\.gz)?$"),
    re.compile(r"^cps_org_clean\.parquet$"),
    re.compile(r"^Three Essays on Immigrants.*\.pdf$", re.IGNORECASE),
    re.compile(r"^essay[123]_audit\.md$", re.IGNORECASE),
    re.compile(r"^\.DS_Store$"),
]

FORBIDDEN_PATH_SUBSTRINGS = [
    "/data/ipums/raw/",
    "/profiling/cache/",
    "/.venv/",
    "/.latex/",
    "/__pycache__/",
    "/.ruff_cache/",
]

FORBIDDEN_TEXT_PATTERNS = [
    (re.compile(r"/Users/[A-Za-z0-9._-]+/"), "Hardcoded macOS user path"),
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), "Possible Google API key"),
    (re.compile(r"ghp_[0-9A-Za-z]{36}"), "Possible GitHub personal access token"),
    (re.compile(r"sk-[0-9A-Za-z]{32,}"), "Possible secret API key"),
]

TEXT_EXTENSIONS = {
    ".py", ".sh", ".md", ".tex", ".typ", ".bib", ".yml", ".yaml",
    ".toml", ".cff", ".txt", ".json", ".csv",
}

REQUIRED_STUDY_YAML_KEYS = {
    "id", "slug", "discipline", "type", "title", "authors",
    "provenance", "date", "status", "license", "papers",
}


def iter_repo_files() -> list[Path]:
    # Inside a git repository, scan tracked + non-ignored files so local
    # gitignored OS artifacts (like macOS .DS_Store) don't trigger false
    # positives while any accidentally committed .DS_Store is still caught.
    if (ROOT / ".git").exists():
        res = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if res.returncode == 0:
            entries = [e for e in res.stdout.decode("utf-8", errors="surrogateescape").split("\0") if e]
            return sorted(ROOT / e for e in entries if (ROOT / e).is_file())

    in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    files: list[Path] = []
    for p in ROOT.rglob("*"):
        if ".git" in p.parts:
            continue
        if not in_ci and p.name == ".DS_Store":
            continue
        if p.is_file():
            files.append(p)
    return sorted(files)


def check_files(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        rel_slash = "/" + rel

        # 1. File size guardrail
        size = p.stat().st_size
        if size > MAX_FILE_BYTES:
            errors.append(f"[SIZE] {rel} is {size / (1024*1024):.2f} MB (> {MAX_FILE_BYTES // (1024*1024)} MB)")

        # 2. Forbidden filenames & paths
        for pat in FORBIDDEN_FILENAME_PATTERNS:
            if pat.search(p.name):
                errors.append(f"[FORBIDDEN FILE] {rel} matches {pat.pattern}")
        for sub in FORBIDDEN_PATH_SUBSTRINGS:
            if sub in rel_slash:
                errors.append(f"[FORBIDDEN PATH] {rel} contains {sub}")
        if "/papers/" in rel_slash and "/results/" in rel_slash and p.suffix == ".parquet":
            errors.append(f"[MICRODATA GUARD] {rel}: parquet files under papers/*/results/ are disallowed")

        # 3. Text scans for paths & secrets (skip self)
        if rel == ".github/scripts/check_hygiene.py":
            continue
        if p.suffix.lower() in TEXT_EXTENSIONS:
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                errors.append(f"[READ ERROR] {rel}: {exc}")
                continue
            for pat, label in FORBIDDEN_TEXT_PATTERNS:
                m = pat.search(content)
                if m:
                    errors.append(f"[CONTENT: {label}] {rel} matched {m.group(0)!r}")
    return errors


def check_study_manifests() -> list[str]:
    errors: list[str] = []
    studies_root = ROOT / "studies"
    if not studies_root.is_dir():
        return ["[MANIFEST] Missing studies/ directory"]

    study_dirs = [
        d for disc in sorted(studies_root.iterdir()) if disc.is_dir()
        for d in sorted(disc.iterdir()) if d.is_dir()
    ]
    if not study_dirs:
        return ["[MANIFEST] No study directories found under studies/<discipline>/<study-slug>/"]

    for sdir in study_dirs:
        rel_dir = sdir.relative_to(ROOT).as_posix()
        for req in ("README.md", "study.yaml", "CITATION.bib"):
            if not (sdir / req).is_file():
                errors.append(f"[MANIFEST] {rel_dir} missing {req}")

        manifest_path = sdir / "study.yaml"
        if manifest_path.is_file() and yaml is not None:
            try:
                data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"[MANIFEST] Failed to parse {rel_dir}/study.yaml: {exc}")
                continue
            if not isinstance(data, dict):
                errors.append(f"[MANIFEST] {rel_dir}/study.yaml is not a YAML mapping")
                continue
            missing = REQUIRED_STUDY_YAML_KEYS - set(data.keys())
            if missing:
                errors.append(f"[MANIFEST] {rel_dir}/study.yaml missing keys: {sorted(missing)}")
            papers = data.get("papers") or []
            if not isinstance(papers, list) or not papers:
                errors.append(f"[MANIFEST] {rel_dir}/study.yaml must define a non-empty papers list")
            for idx, paper in enumerate(papers):
                for path_key in ("pdf", "source"):
                    rel_art = paper.get(path_key)
                    if rel_art and not (sdir / rel_art).is_file():
                        errors.append(f"[MANIFEST] {rel_dir}/study.yaml papers[{idx}].{path_key} missing: {rel_art}")
    return errors


def main() -> int:
    files = iter_repo_files()
    errors = check_files(files) + check_study_manifests()
    total_mb = sum(p.stat().st_size for p in files) / (1024 * 1024)
    print(f"Scanned {len(files)} files ({total_mb:.2f} MB total).")
    if errors:
        print(f"\nFAILED with {len(errors)} issue(s):")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("PASSED: All repository hygiene, privacy, size, and study.yaml checks succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
