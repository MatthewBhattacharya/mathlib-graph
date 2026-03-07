#!/usr/bin/env python3
"""
Build a complete dependency database for Mathlib.Analysis.

Fetches the full .lean source files from GitHub (which include proof bodies
that corpus.jsonl omits for tactic-mode theorems), then extracts dependency
edges from every line of every declaration — covering both tactic-mode and
term-mode proofs.

The result is written to  data/analysis_full.db  (a fresh DuckDB file).
The main DB (data/mathlib.db) is only read, never modified.

Edge sources in the output DB
------------------------------
  1. Tactic-trace edges   – copied directly from main DB (precise, from LeanDojo)
  2. Full-source edges     – new, extracted here from GitHub source (both proof styles)
     For each declaration the FULL span (start_line → end_line) is scanned,
     so term-mode proof bodies and type-signature references are both captured.
"""

import re
import time
import duckdb
import requests
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT    = Path(__file__).parent.parent
MAIN_DB = ROOT / "data" / "mathlib.db"
OUT_DB  = ROOT / "data" / "analysis_full.db"
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


# ─────────────────────────────────────────────────────────────────────────────

def fetch_file(session: requests.Session, path: str) -> list[str] | None:
    """
    Fetch a raw .lean file from GitHub.
    Returns a list where index i contains line i (1-indexed); index 0 is unused.
    """
    url = RAW_BASE + path
    try:
        resp = session.get(url, timeout=20)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    # Prepend a dummy entry so lines[n] == line n (1-based)
    return [""] + resp.text.splitlines()


def refs_in_span(
    lines: list[str], start: int, end: int, known: frozenset[str]
) -> list[str]:
    """Return known declaration names referenced in lines[start..end] (inclusive, 1-based)."""
    text = "\n".join(lines[start : end + 1])
    return [m for m in _QUALIFIED_RE.findall(text) if m in known]


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Connect ───────────────────────────────────────────────────────────────
    src = duckdb.connect(str(MAIN_DB), read_only=True)

    if OUT_DB.exists():
        OUT_DB.unlink()
    out = duckdb.connect(str(OUT_DB))

    # Create schema without FK constraints on dependencies
    # (dst may point to non-Analysis declarations from the wider Mathlib)
    out.execute("""
        CREATE TABLE declarations (
            name         TEXT PRIMARY KEY,
            module       TEXT,
            module_parts TEXT[],
            kind         TEXT,
            type_sig     TEXT,
            in_leandojo  BOOL DEFAULT FALSE,
            in_mltypes   BOOL DEFAULT FALSE
        );

        CREATE TABLE dependencies (
            src  TEXT NOT NULL,
            dst  TEXT NOT NULL,
            source TEXT,           -- 'tactic' | 'source_scan'
            PRIMARY KEY (src, dst)
        );

        CREATE INDEX idx_dep_src ON dependencies(src);
        CREATE INDEX idx_dep_dst ON dependencies(dst);
        CREATE INDEX idx_decl_mod ON declarations(module);
    """)

    # ── Load all declaration names (for ref matching) ─────────────────────────
    print("Loading declaration names from main DB …")
    all_names: frozenset[str] = frozenset(
        r[0] for r in src.execute("SELECT name FROM declarations").fetchall()
    )
    print(f"  {len(all_names):,} names")

    # ── Analysis declarations ─────────────────────────────────────────────────
    print("\nLoading Analysis declarations …")
    analysis_rows = src.execute("""
        SELECT d.name, d.module, d.module_parts, d.kind, d.type_sig,
               d.in_leandojo, d.in_mltypes,
               ls.file_path, ls.start_line, ls.end_line
        FROM declarations d
        JOIN lean_source ls ON d.name = ls.name
        WHERE d.module LIKE 'Mathlib.Analysis%'
    """).fetchall()
    print(f"  {len(analysis_rows):,} declarations")

    # Insert into output DB
    out.executemany(
        "INSERT OR IGNORE INTO declarations "
        "(name, module, module_parts, kind, type_sig, in_leandojo, in_mltypes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [r[:7] for r in analysis_rows],
    )

    # Group by file for GitHub fetch
    by_file: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for row in analysis_rows:
        name, *_, fp, start, end = row
        if fp and start and end:
            by_file[fp].append((name, start, end))

    print(f"  {len(by_file):,} unique files to fetch")

    # ── Copy existing tactic-trace edges from main DB ─────────────────────────
    print("\nCopying tactic-trace edges from main DB …")
    tactic_edges: list[tuple[str, str]] = src.execute("""
        SELECT DISTINCT dep.src, dep.dst
        FROM dependencies dep
        JOIN declarations d ON d.name = dep.src
        WHERE d.module LIKE 'Mathlib.Analysis%'
    """).fetchall()
    print(f"  {len(tactic_edges):,} tactic edges")

    out.executemany(
        "INSERT OR IGNORE INTO dependencies (src, dst, source) VALUES (?, ?, 'tactic')",
        tactic_edges,
    )

    existing_edges: frozenset[tuple[str, str]] = frozenset(tactic_edges)

    # ── Fetch .lean files from GitHub ─────────────────────────────────────────
    print(f"\nFetching {len(by_file):,} .lean files from GitHub …")
    session = requests.Session()
    session.headers["User-Agent"] = "mathlib-atlas/1.0"

    file_cache: dict[str, list[str]] = {}
    fetch_errors = 0

    for fp in tqdm(list(by_file.keys()), unit="file", smoothing=0.1):
        lines = fetch_file(session, fp)
        if lines is not None:
            file_cache[fp] = lines
        else:
            fetch_errors += 1
        time.sleep(0.05)   # ~20 req/s — polite, stays under GitHub limits

    print(f"  Fetched {len(file_cache):,} files  ({fetch_errors} errors)")

    # ── Extract full-source dependency edges ──────────────────────────────────
    print("\nExtracting dependencies from full declaration spans …")

    new_edges: set[tuple[str, str]] = set()
    for fp, decl_list in tqdm(by_file.items(), unit="file", smoothing=0.1):
        lines = file_cache.get(fp)
        if lines is None:
            continue
        for name, start, end in decl_list:
            for ref in refs_in_span(lines, start, end, all_names):
                if ref != name:
                    edge = (name, ref)
                    if edge not in existing_edges:
                        new_edges.add(edge)

    print(f"  {len(new_edges):,} new edges (beyond tactic trace)")

    # ── Insert new edges ──────────────────────────────────────────────────────
    print("Inserting source-scan edges …")
    out.executemany(
        "INSERT OR IGNORE INTO dependencies (src, dst, source) VALUES (?, ?, 'source_scan')",
        [(s, d) for s, d in new_edges],
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_decl  = out.sql("SELECT COUNT(*) FROM declarations").fetchone()[0]
    n_total = out.sql("SELECT COUNT(*) FROM dependencies").fetchone()[0]
    n_tac   = out.sql("SELECT COUNT(*) FROM dependencies WHERE source='tactic'").fetchone()[0]
    n_src   = out.sql("SELECT COUNT(*) FROM dependencies WHERE source='source_scan'").fetchone()[0]

    # Nodes that now have at least one outgoing edge
    n_connected = out.sql(
        "SELECT COUNT(DISTINCT src) FROM dependencies"
    ).fetchone()[0]
    n_isolated = n_decl - n_connected

    print(f"""
╔═══════════════════════════════════════════════════╗
║  analysis_full.db  summary                        ║
╠═══════════════════════════════════════════════════╣
║  Declarations  : {n_decl:>10,}                      ║
║  Total edges   : {n_total:>10,}                      ║
║    tactic trace: {n_tac:>10,}                      ║
║    source scan : {n_src:>10,}                      ║
║  Nodes w/ edges: {n_connected:>10,}                      ║
║  Isolated nodes: {n_isolated:>10,}                      ║
╚═══════════════════════════════════════════════════╝
Output: {OUT_DB}""")

    src.close()
    out.close()


if __name__ == "__main__":
    main()
