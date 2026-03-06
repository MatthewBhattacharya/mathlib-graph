#!/usr/bin/env python3
"""
Enrich mathlib.db with dependency edges extracted from full Lean source files.

For each declaration in lean_source we have (file_path, start_line, end_line).
corpus.jsonl stored only the type signature in the 'code' field for tactic-mode
proofs, so proof bodies were missing.  This script fetches the raw .lean files
from GitHub and scans the complete span (docstring + signature + proof body)
for every declaration, adding any qualified-name references that are missing
from the dependencies table.

Edges added here complement the tactic-trace edges already in the table.
No existing data is deleted or modified.
"""

import re
import time
import duckdb
import requests
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "mathlib.db"
COMMIT  = "29dcec074de168ac2bf835a77ef68bbe069194c5"
RAW_BASE = (
    f"https://raw.githubusercontent.com/leanprover-community/mathlib4/{COMMIT}/"
)

# ── Lean 4 qualified-name regex ───────────────────────────────────────────────
_IDENT = (
    r'[A-Za-z_'
    r'\xc0-\u024f'       # Latin Extended A/B
    r'\u0370-\u03ff'     # Greek (α β γ …)
    r'\u1d00-\u1dbf'     # Phonetic extensions
    r'\u2100-\u214f'     # Letterlike symbols (ℕ ℤ ℝ …)
    r'][A-Za-z0-9_\''
    r'\xc0-\u024f'
    r'\u0370-\u03ff'
    r'\u1d00-\u1dbf'
    r'\u2100-\u214f'
    r']*'
)
_QUALIFIED_RE = re.compile(rf'(?:{_IDENT}\.)+{_IDENT}')


def fetch_file(session: requests.Session, path: str) -> list[str] | None:
    """
    Fetch a raw .lean file.  Returns a list indexed from 1 (lines[0] unused).
    Returns None on any HTTP or network error.
    """
    try:
        resp = session.get(RAW_BASE + path, timeout=30)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    return [""] + resp.text.splitlines()   # lines[n] == line n (1-based)


def refs_in_span(
    lines: list[str], start: int, end: int, known: frozenset[str]
) -> list[str]:
    text = "\n".join(lines[start : end + 1])
    return [m for m in _QUALIFIED_RE.findall(text) if m in known]


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    db = duckdb.connect(str(DB_PATH))

    # ── Load lookup structures ────────────────────────────────────────────────
    print("Loading declaration names …")
    all_names: frozenset[str] = frozenset(
        r[0] for r in db.execute("SELECT name FROM declarations").fetchall()
    )
    print(f"  {len(all_names):,} names")

    print("Loading existing edges …")
    existing_edges: frozenset[tuple[str, str]] = frozenset(
        (r[0], r[1]) for r in db.execute("SELECT src, dst FROM dependencies").fetchall()
    )
    print(f"  {len(existing_edges):,} existing edges")

    # ── Gather all (file_path, decl_name, start_line, end_line) ──────────────
    print("\nLoading lean_source entries …")
    source_rows = db.execute("""
        SELECT ls.name, ls.file_path, ls.start_line, ls.end_line
        FROM lean_source ls
        JOIN declarations d ON d.name = ls.name
        WHERE d.module LIKE 'Mathlib%'
          AND ls.file_path IS NOT NULL
          AND ls.start_line IS NOT NULL
          AND ls.end_line   IS NOT NULL
    """).fetchall()
    print(f"  {len(source_rows):,} declarations with file positions")

    # Group by file so we fetch each file exactly once
    by_file: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for name, fp, start, end in source_rows:
        by_file[fp].append((name, start, end))
    print(f"  {len(by_file):,} unique files to fetch")

    # ── Fetch files from GitHub ───────────────────────────────────────────────
    print(f"\nFetching {len(by_file):,} .lean files from GitHub …")
    session = requests.Session()
    session.headers["User-Agent"] = "mathlib-metaanalysis/1.0"

    file_cache: dict[str, list[str]] = {}
    fetch_errors = 0

    for fp in tqdm(list(by_file.keys()), unit="file", smoothing=0.05):
        lines = fetch_file(session, fp)
        if lines is not None:
            file_cache[fp] = lines
        else:
            fetch_errors += 1
        time.sleep(0.05)   # ~20 req/s

    print(f"  OK: {len(file_cache):,}   errors: {fetch_errors}")

    # ── Extract new dependency edges ──────────────────────────────────────────
    print("\nExtracting edges from full declaration spans …")
    new_edges: set[tuple[str, str]] = set()

    for fp, decl_list in tqdm(by_file.items(), unit="file", smoothing=0.05):
        lines = file_cache.get(fp)
        if lines is None:
            continue
        for name, start, end in decl_list:
            for ref in refs_in_span(lines, start, end, all_names):
                if ref != name:
                    edge = (name, ref)
                    if edge not in existing_edges:
                        new_edges.add(edge)

    print(f"  {len(new_edges):,} new edges to insert")

    # ── Bulk-insert new edges (pre-filtered → no OR IGNORE overhead) ──────────
    # We already know every (src, dst) is valid (both in all_names, not duplicate)
    # so plain INSERT is safe and much faster than INSERT OR IGNORE.
    print("Inserting …")
    edge_list = list(new_edges)
    BATCH = 50_000
    for i in tqdm(range(0, len(edge_list), BATCH), unit="batch"):
        batch = edge_list[i : i + BATCH]
        db.executemany("INSERT INTO dependencies (src, dst) VALUES (?, ?)", batch)

    final = db.execute("SELECT COUNT(*) FROM dependencies").fetchone()[0]
    added = final - len(existing_edges)

    print(f"""
╔══════════════════════════════════════════════════╗
║  mathlib.db  enriched with full-source edges     ║
╠══════════════════════════════════════════════════╣
║  Files fetched  : {len(file_cache):>8,}                    ║
║  Fetch errors   : {fetch_errors:>8,}                    ║
║  New edges added: {added:>8,}                    ║
║  Total edges    : {final:>8,}                    ║
╚══════════════════════════════════════════════════╝""")

    db.close()


if __name__ == "__main__":
    main()
