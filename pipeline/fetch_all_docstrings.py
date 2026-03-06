#!/usr/bin/env python3
"""
Fetch docstrings for all Mathlib declarations from GitHub source.

Uses the lean_source table (already in mathlib.db) for file positions —
no need to re-scan corpus.jsonl.  The start_line for each declaration
points to the /-- docstring block when one is present.

Safe to re-run: declarations already in the docstrings table are skipped.
"""

import time
import duckdb
import requests
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "mathlib.db"
COMMIT  = "29dcec074de168ac2bf835a77ef68bbe069194c5"
RAW_BASE = f"https://raw.githubusercontent.com/leanprover-community/mathlib4/{COMMIT}/"

BATCH_FLUSH = 2_000   # INSERT after this many new docstrings accumulate


def extract_docstring(lines: list[str], start_line: int) -> str | None:
    """
    Extract a /-- ... -/ docstring starting at start_line (1-indexed).
    Returns the inner text, or None if no docstring is present.
    """
    idx = start_line - 1          # convert to 0-indexed
    if idx < 0 or idx >= len(lines):
        return None

    first = lines[idx].lstrip()
    if not first.startswith("/--"):
        return None

    # Single-line:  /-- text -/
    if "-/" in first:
        inner = first[3 : first.index("-/")].strip()
        return inner or None

    # Multi-line: collect until the closing -/
    parts = [first[3:].rstrip()]
    for i in range(idx + 1, len(lines)):
        raw = lines[i]
        if "-/" in raw:
            parts.append(raw[: raw.index("-/")].rstrip())
            break
        parts.append(raw.rstrip())

    return "\n".join(parts).strip() or None


def main() -> None:
    db = duckdb.connect(str(DB_PATH))

    # ── Load all lean_source entries with file positions ──────────────────
    print("Loading lean_source entries …")
    source_rows = db.execute("""
        SELECT ls.name, ls.file_path, ls.start_line
        FROM lean_source ls
        JOIN declarations d ON d.name = ls.name
        WHERE d.module LIKE 'Mathlib%'
          AND ls.file_path  IS NOT NULL
          AND ls.start_line IS NOT NULL
    """).fetchall()
    print(f"  {len(source_rows):,} declarations with file positions")

    # Skip declarations already in the docstrings table
    existing: frozenset[str] = frozenset(
        r[0] for r in db.execute("SELECT name FROM docstrings").fetchall()
    )
    print(f"  {len(existing):,} already have docstrings — skipping")

    # Group remaining work by file
    by_file: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for name, fp, start in source_rows:
        if name not in existing:
            by_file[fp].append((name, start))

    n_todo = sum(len(v) for v in by_file.values())
    print(f"  {n_todo:,} declarations across {len(by_file):,} files to check")

    # ── Fetch files and extract docstrings ────────────────────────────────
    session = requests.Session()
    session.headers["User-Agent"] = "mathlib-metaanalysis/1.0"

    pending: list[tuple[str, str]] = []
    fetch_errors = 0
    n_found = 0

    def flush():
        if pending:
            db.executemany(
                "INSERT OR REPLACE INTO docstrings (name, docstring) VALUES (?, ?)",
                pending,
            )
            pending.clear()

    for fp in tqdm(list(by_file.keys()), unit="file", smoothing=0.05):
        try:
            resp = session.get(RAW_BASE + fp, timeout=30)
        except requests.RequestException:
            fetch_errors += 1
            time.sleep(0.05)
            continue

        if resp.status_code != 200:
            fetch_errors += 1
            time.sleep(0.05)
            continue

        lines = resp.text.splitlines()

        for name, start in by_file[fp]:
            doc = extract_docstring(lines, start)
            if doc:
                pending.append((name, doc))
                n_found += 1

        time.sleep(0.05)   # ~20 req/s — polite rate

        if len(pending) >= BATCH_FLUSH:
            flush()

    flush()

    # ── Summary ───────────────────────────────────────────────────────────
    total_docs = db.execute("SELECT COUNT(*) FROM docstrings").fetchone()[0]
    named = db.execute(r"""
        SELECT COUNT(*) FROM docstrings
        WHERE regexp_matches(docstring, '\*\*[^*]+\*\*')
    """).fetchone()[0]

    # Named by top-level module area
    by_area = db.execute(r"""
        SELECT d.module_parts[2] AS area,
               COUNT(*)          AS n_named
        FROM docstrings ds
        JOIN declarations d ON d.name = ds.name
        WHERE regexp_matches(ds.docstring, '\*\*[^*]+\*\*')
          AND d.module_parts[1] = 'Mathlib'
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 20
    """).fetchall()

    print(f"""
╔══════════════════════════════════════════════════╗
║  Docstring fetch complete                        ║
╠══════════════════════════════════════════════════╣
║  New docstrings found : {n_found:>8,}                    ║
║  Fetch errors         : {fetch_errors:>8,}                    ║
║  Total in docstrings  : {total_docs:>8,}                    ║
║  Named theorems (bold): {named:>8,}                    ║
╚══════════════════════════════════════════════════╝

Named theorems by area:""")
    for area, n in by_area:
        print(f"  {str(area):30s}  {n:4,}")

    db.close()


if __name__ == "__main__":
    main()
