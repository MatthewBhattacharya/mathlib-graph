"""
fetch_docstrings.py — Fetch docstrings for Mathlib.LinearAlgebra declarations from GitHub.

How it works:
  corpus.jsonl gives us (file_path, start_line) for each declaration.
  The start_line points to the /-- docstring -/ block when one exists,
  or directly to the declaration keyword when it doesn't.
  We fetch each source file once from GitHub and extract docstrings by position.

Writes to: data/mathlib.db → docstrings table
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import polars as pl
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "mathlib.db"
CORPUS_PATH = ROOT / "data" / "raw" / "leandojo" / "corpus.jsonl"

COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"
GITHUB_RAW = (
    f"https://raw.githubusercontent.com/leanprover-community/mathlib4/{COMMIT}"
)

# Change this to restrict to a different Mathlib area
MODULE_FILTER = "Mathlib/LinearAlgebra"


# ---------------------------------------------------------------------------
# Docstring extraction
# ---------------------------------------------------------------------------

def extract_docstring(lines: list[str], start_line: int) -> str | None:
    """
    Extract the /-- ... -/ docstring at start_line (1-indexed).

    In Lean 4 source, corpus.jsonl start positions point to the docstring
    line when one is present, or directly to the declaration otherwise.

    Returns the inner text (delimiters stripped), or None if no docstring.
    """
    idx = start_line - 1  # 0-indexed
    if idx < 0 or idx >= len(lines):
        return None

    first = lines[idx].lstrip()
    if not first.startswith("/--"):
        return None

    # Single-line: /-- text -/
    if "-/" in first:
        end = first.index("-/")
        inner = first[3:end].strip()
        return inner or None

    # Multi-line: collect until the line containing -/
    parts = [first[3:].rstrip()]
    for i in range(idx + 1, len(lines)):
        raw = lines[i]
        if "-/" in raw:
            parts.append(raw[: raw.index("-/")].rstrip())
            break
        parts.append(raw.rstrip())

    text = "\n".join(parts).strip()
    return text or None


# ---------------------------------------------------------------------------
# GitHub fetching
# ---------------------------------------------------------------------------

def fetch_raw(file_path: str, session: requests.Session) -> list[str] | None:
    """Fetch a file from GitHub raw content. Returns lines or None on 404."""
    url = f"{GITHUB_RAW}/{file_path}"
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.text.splitlines()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(module_filter: str = MODULE_FILTER) -> None:
    # --- Collect (file_path → {name: start_line}) from corpus.jsonl ---
    print(f"[docstrings] Scanning corpus.jsonl for '{module_filter}' declarations …")

    file_decls: dict[str, dict[str, int]] = defaultdict(dict)

    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            path = entry.get("path", "")
            if module_filter not in path:
                continue
            for prem in entry.get("premises", []):
                name = prem.get("full_name", "")
                start = prem.get("start", [])
                if name and start:
                    file_decls[path][name] = start[0]  # 1-indexed

    n_files = len(file_decls)
    n_decls = sum(len(v) for v in file_decls.values())
    print(f"[docstrings] {n_files} files, {n_decls:,} declarations to check")

    # --- Ensure docstrings table exists ---
    db = duckdb.connect(str(DB_PATH))
    db.execute("""
        CREATE TABLE IF NOT EXISTS docstrings (
            name      TEXT PRIMARY KEY,
            docstring TEXT NOT NULL
        )
    """)

    # --- Fetch and extract ---
    session = requests.Session()
    session.headers["User-Agent"] = "mathlib-metaanalysis/1.0"

    rows: list[dict] = []
    missing_files = []

    for file_path, decls in tqdm(file_decls.items(), desc="fetching files", unit="file"):
        lines = fetch_raw(file_path, session)
        if lines is None:
            missing_files.append(file_path)
            continue

        for name, start_line in decls.items():
            doc = extract_docstring(lines, start_line)
            if doc:
                rows.append({"name": name, "docstring": doc})

        time.sleep(0.05)  # ~20 req/s

    # --- Store results ---
    if rows:
        df = pl.DataFrame(rows, schema={"name": pl.String, "docstring": pl.String})
        db.register("doc_batch", df.to_arrow())
        db.execute("INSERT OR REPLACE INTO docstrings SELECT * FROM doc_batch")
        db.unregister("doc_batch")

    db.close()

    n_found = len(rows)
    pct = 100 * n_found / n_decls if n_decls else 0
    print(f"\n[docstrings] {n_found:,} / {n_decls:,} declarations have docstrings ({pct:.1f}%)")
    if missing_files:
        print(f"  {len(missing_files)} files returned 404 (may have been moved in this commit)")


if __name__ == "__main__":
    main()
