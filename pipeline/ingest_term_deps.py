#!/usr/bin/env python3
"""
Extract dependency edges from term-mode proofs and type signatures by scanning
Lean source code for references to known declarations.

This complements the tactic-trace edges already in the dependencies table,
which only cover tactic-mode proofs (~62k theorems).  The ~119k declarations
without tactic coverage (term-mode proofs, defs, instances, abbrevs, etc.)
get no edges from tactic tracing, so this script fills that gap.

Strategy
--------
For declarations WITHOUT tactic edges (term-mode):
    Scan the entire code block for qualified identifiers (Foo.bar.baz style).
    Any that match a known declaration name become dependency edges.

For declarations WITH tactic edges (tactic-mode):
    Scan only the TYPE SIGNATURE (code up to ':=') for qualified identifiers.
    The proof body is already covered by LeanDojo tactic tracing.

False positives are possible (e.g. 'hp.one_lt' on a local variable 'hp') but
are filtered naturally because such pseudo-names won't appear in the
declarations table.
"""

from pathlib import Path
import re
import duckdb
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "mathlib.db"

# ---------------------------------------------------------------------------
# Lean 4 qualified-name regex
# A qualified name is two or more identifier components separated by dots.
# Components can contain ASCII letters, digits, _, ' and common math Unicode.
# We deliberately exclude single-component names to reduce noise (they'd need
# open-namespace resolution to disambiguate, which we don't attempt here).
# ---------------------------------------------------------------------------
_IDENT = (
    r'[A-Za-z_'
    r'\xc0-\u024f'      # Latin Extended A/B
    r'\u0370-\u03ff'    # Greek
    r'\u1d00-\u1dbf'    # Phonetic extensions
    r'\u2100-\u214f'    # Letterlike symbols (ℕ, ℤ, ℝ, …)
    r'][A-Za-z0-9_\''
    r'\xc0-\u024f'
    r'\u0370-\u03ff'
    r'\u1d00-\u1dbf'
    r'\u2100-\u214f'
    r']*'
)
_QUALIFIED_RE = re.compile(rf'(?:{_IDENT}\.)+{_IDENT}')


def sig_only(code: str) -> str:
    """Return the declaration header (before ':='), or the whole code if no ':='."""
    idx = code.find(':=')
    return code[:idx] if idx >= 0 else code


def extract_refs(code: str, known: set[str]) -> list[str]:
    """Find all qualified identifiers in *code* that are in *known*."""
    return [m for m in _QUALIFIED_RE.findall(code) if m in known]


def main() -> None:
    db = duckdb.connect(str(DB_PATH))

    # ------------------------------------------------------------------
    # 1. Load all known declaration names into a Python set for O(1) lookup
    # ------------------------------------------------------------------
    print("Loading declaration names …")
    all_names: set[str] = set(
        r[0] for r in db.execute("SELECT name FROM declarations").fetchall()
    )
    print(f"  {len(all_names):,} names")

    # ------------------------------------------------------------------
    # 2. Fetch lean_source entries, split by tactic coverage
    # ------------------------------------------------------------------
    print("\nFetching lean_source entries …")

    # Term-mode: no entry in tactics table → scan full code
    term_rows = db.execute("""
        SELECT ls.name, ls.code
        FROM lean_source ls
        JOIN declarations d ON d.name = ls.name
        LEFT JOIN (SELECT DISTINCT theorem_name FROM tactics) t
               ON ls.name = t.theorem_name
        WHERE t.theorem_name IS NULL
          AND ls.code IS NOT NULL
    """).fetchall()
    print(f"  {len(term_rows):,}  term-mode / non-tactic declarations  (full code scan)")

    # Tactic-mode: scan type signature only (proof body covered by tactic trace)
    tactic_rows = db.execute("""
        SELECT ls.name, ls.code
        FROM lean_source ls
        JOIN declarations d ON d.name = ls.name
        JOIN (SELECT DISTINCT theorem_name FROM tactics) t
          ON ls.name = t.theorem_name
        WHERE ls.code IS NOT NULL
    """).fetchall()
    print(f"  {len(tactic_rows):,}  tactic-mode declarations         (signature scan only)")

    # ------------------------------------------------------------------
    # 3. Scan for qualified-name references
    # ------------------------------------------------------------------
    new_edges: set[tuple[str, str]] = set()

    print("\nScanning term-mode declarations (full code) …")
    for name, code in tqdm(term_rows, unit="decl", smoothing=0.05):
        for ref in extract_refs(code, all_names):
            if ref != name:
                new_edges.add((name, ref))

    n_term = len(new_edges)
    print(f"  → {n_term:,} candidate edges from term-mode code")

    print("\nScanning tactic-mode type signatures …")
    for name, code in tqdm(tactic_rows, unit="decl", smoothing=0.05):
        for ref in extract_refs(sig_only(code), all_names):
            if ref != name:
                new_edges.add((name, ref))

    n_sig = len(new_edges) - n_term
    print(f"  → {n_sig:,} additional edges from tactic-mode type signatures")
    print(f"  → {len(new_edges):,} total candidate edges to insert")

    # ------------------------------------------------------------------
    # 4. Insert into dependencies (skip duplicates and FK violations)
    # ------------------------------------------------------------------
    existing = db.execute("SELECT COUNT(*) FROM dependencies").fetchone()[0]
    print(f"\nExisting edges in dependencies: {existing:,}")
    print("Inserting new edges …")

    edge_list = list(new_edges)
    BATCH = 10_000
    errors = 0
    for i in tqdm(range(0, len(edge_list), BATCH), unit="batch"):
        batch = edge_list[i : i + BATCH]
        try:
            db.executemany(
                "INSERT OR IGNORE INTO dependencies (src, dst) VALUES (?, ?)",
                batch,
            )
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Warning (batch {i // BATCH}): {e}")

    final = db.execute("SELECT COUNT(*) FROM dependencies").fetchone()[0]
    added = final - existing
    print(f"\nDone.")
    print(f"  Dependencies: {existing:,} → {final:,}  (+{added:,} new edges)")
    if errors:
        print(f"  Batches with errors: {errors}")

    db.close()


if __name__ == "__main__":
    main()
