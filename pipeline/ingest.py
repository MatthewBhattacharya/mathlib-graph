"""
ingest.py — Parse raw datasets and load into DuckDB.

Reads:
  data/raw/mathlib_types/             ← Parquet shards (name, module, type)
  data/raw/leandojo/corpus.jsonl      ← per-file: {path, imports, premises:[{full_name, code, kind}]}
  data/raw/leandojo/leandojo_benchmark_4/{random,novel_premises}/{train,val,test}.json
                                      ← per-theorem: {full_name, traced_tactics:[{annotated_tactic}]}

Writes:
  data/mathlib.db  ← DuckDB with `declarations` + `dependencies` tables
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterator

import duckdb
import polars as pl
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "mathlib.db"
SCHEMA_SQL = Path(__file__).parent / "schema.sql"
MLTYPES_DIR = ROOT / "data" / "raw" / "mathlib_types"
LEANDOJO_DIR = ROOT / "data" / "raw" / "leandojo"
CORPUS_PATH = LEANDOJO_DIR / "corpus.jsonl"
BENCHMARK_DIR = LEANDOJO_DIR / "leandojo_benchmark_4"

# All theorem split files (random + novel_premises, train + val + test)
SPLIT_GLOBS = ["random/*.json", "novel_premises/*.json"]


# ---------------------------------------------------------------------------
# Kind inference
# ---------------------------------------------------------------------------

# Lean declaration keywords, possibly after attributes/modifiers
_DECL_KEYWORDS = re.compile(
    r"^(?:@\[[^\]]*\]\s*)*"          # zero or more @[...] attributes
    r"(?:private\s+|protected\s+|noncomputable\s+|unsafe\s+)*"  # modifiers
    r"(theorem|lemma|def|definition|instance|axiom|abbrev|class|structure|inductive)\b",
    re.MULTILINE,
)


def kind_from_code(code: str) -> str:
    """Infer declaration kind from raw Lean source code snippet."""
    m = _DECL_KEYWORDS.match(code.strip())
    if not m:
        return "unknown"
    kw = m.group(1)
    if kw in ("theorem", "lemma"):
        return kw
    if kw in ("def", "definition", "abbrev"):
        return "def"
    if kw == "instance":
        return "instance"
    if kw == "axiom":
        return "axiom"
    if kw in ("class", "structure", "inductive"):
        return "def"
    return "unknown"


def infer_kind(name: str, type_sig: str) -> str:
    """Heuristic kind from mathlib-types type signature (less reliable than code)."""
    if not type_sig:
        return "unknown"
    sig = type_sig.strip()
    first = sig.split()[0].lower() if sig.split() else ""
    if first in ("theorem", "lemma"):
        return first
    if first in ("def", "definition", "noncomputable"):
        return "def"
    if first == "instance":
        return "instance"
    if first == "axiom":
        return "axiom"
    if re.search(r"\bProp\b", sig):
        return "theorem"
    if re.search(r"Instance$|\.inst", name):
        return "instance"
    return "unknown"


def _module_from_filepath(file_path: str) -> str | None:
    """Derive module name from a Lean file path."""
    p = Path(file_path)
    parts = list(p.parts)
    try:
        idx = next(i for i, part in enumerate(parts) if part in ("Mathlib", "lake-packages"))
        parts = parts[idx:]
    except StopIteration:
        pass
    if parts and parts[-1].endswith(".lean"):
        parts[-1] = parts[-1][:-5]
    return ".".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Step 1: Load mathlib-types Parquet → declarations
# ---------------------------------------------------------------------------

def ingest_mathlib_types(db: duckdb.DuckDBPyConnection) -> int:
    parquet_files = sorted(MLTYPES_DIR.rglob("*.parquet"))
    if not parquet_files:
        print(f"[mltypes] No Parquet files found in {MLTYPES_DIR}", file=sys.stderr)
        return 0

    print(f"[mltypes] Loading {len(parquet_files)} Parquet shard(s) …")
    df = pl.read_parquet([str(p) for p in parquet_files])
    print(f"[mltypes] Raw rows: {len(df):,}")

    col_map = {}
    for col in df.columns:
        if col.lower() in ("type", "type_sig", "signature"):
            col_map[col] = "type_sig"
        elif col.lower() in ("name", "full_name"):
            col_map[col] = "name"
        elif col.lower() == "module":
            col_map[col] = "module"
    df = df.rename(col_map)

    for c in ("name", "module", "type_sig"):
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))

    df = df.select(["name", "module", "type_sig"])
    df = df.filter(pl.col("name").is_not_null() & (pl.col("name") != ""))

    print("[mltypes] Inferring kinds …")
    df = df.with_columns(
        pl.struct(["name", "type_sig"])
        .map_elements(
            lambda row: infer_kind(row["name"] or "", row["type_sig"] or ""),
            return_dtype=pl.String,
        )
        .alias("kind")
    )
    df = df.with_columns(
        pl.col("module")
        .map_elements(
            lambda m: m.split(".") if m else [],
            return_dtype=pl.List(pl.String),
        )
        .alias("module_parts")
    )
    df = df.with_columns([
        pl.lit(False).alias("in_leandojo"),
        pl.lit(True).alias("in_mltypes"),
    ])
    df = df.select(["name", "module", "module_parts", "kind", "type_sig", "in_leandojo", "in_mltypes"])

    db.register("mltypes_staging", df.to_arrow())
    inserted = db.execute("""
        INSERT OR IGNORE INTO declarations
        SELECT name, module, module_parts, kind, type_sig, in_leandojo, in_mltypes
        FROM mltypes_staging
    """).rowcount
    db.execute("UPDATE declarations SET in_mltypes = TRUE WHERE name IN (SELECT name FROM mltypes_staging)")
    db.unregister("mltypes_staging")
    print(f"[mltypes] Inserted {inserted:,} new declarations")
    return inserted


# ---------------------------------------------------------------------------
# Step 2a: corpus.jsonl → declarations (better kind via code field)
# ---------------------------------------------------------------------------

def ingest_corpus_declarations(db: duckdb.DuckDBPyConnection) -> int:
    """
    Parse corpus.jsonl to extract declarations with kind inferred from code.
    Each line: {path, imports, premises:[{full_name, code, kind, start, end}]}
    """
    if not CORPUS_PATH.exists():
        print(f"[corpus] corpus.jsonl not found at {CORPUS_PATH}", file=sys.stderr)
        return 0

    size_mb = CORPUS_PATH.stat().st_size / 1e6
    print(f"[corpus] Streaming corpus.jsonl ({size_mb:.1f} MB) for declarations …")

    # path → module mapping (built as we scan)
    decls: dict[str, dict] = {}  # full_name → {module, kind, code}

    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in tqdm(f, desc="corpus.jsonl", unit="files"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            file_path = entry.get("path", "")
            module = _module_from_filepath(file_path) if file_path else None

            for prem in entry.get("premises", []):
                full_name = prem.get("full_name", "")
                if not full_name:
                    continue
                code = prem.get("code", "")
                kind = kind_from_code(code)
                if full_name not in decls:
                    decls[full_name] = {"module": module, "kind": kind}
                elif kind != "unknown" and decls[full_name]["kind"] == "unknown":
                    decls[full_name]["kind"] = kind

    print(f"[corpus] Found {len(decls):,} unique declarations")

    rows = [
        {
            "name": name,
            "module": info["module"],
            "module_parts": info["module"].split(".") if info["module"] else [],
            "kind": info["kind"],
            "type_sig": None,
            "in_leandojo": True,
            "in_mltypes": False,
        }
        for name, info in decls.items()
    ]

    df = pl.DataFrame(
        rows,
        schema={
            "name": pl.String,
            "module": pl.String,
            "module_parts": pl.List(pl.String),
            "kind": pl.String,
            "type_sig": pl.String,
            "in_leandojo": pl.Boolean,
            "in_mltypes": pl.Boolean,
        },
    )
    db.register("corpus_decls", df.to_arrow())

    inserted = db.execute("""
        INSERT OR IGNORE INTO declarations
            (name, module, module_parts, kind, type_sig, in_leandojo, in_mltypes)
        SELECT name, module, module_parts, kind, type_sig, in_leandojo, in_mltypes
        FROM corpus_decls
    """).rowcount

    # For existing mltypes rows: mark in_leandojo=TRUE and upgrade kind if we have better info
    db.execute("""
        UPDATE declarations
        SET
            in_leandojo = TRUE,
            module = COALESCE(declarations.module, corpus_decls.module),
            module_parts = COALESCE(declarations.module_parts, corpus_decls.module_parts),
            kind = CASE
                WHEN declarations.kind = 'unknown' AND corpus_decls.kind != 'unknown'
                THEN corpus_decls.kind
                ELSE declarations.kind
            END
        FROM corpus_decls
        WHERE declarations.name = corpus_decls.name
    """)

    db.unregister("corpus_decls")
    print(f"[corpus] Inserted {inserted:,} new declarations (in_leandojo=TRUE)")
    return inserted


# ---------------------------------------------------------------------------
# Step 2b: Split JSONs → dependency edges via annotated_tactic[1]
# ---------------------------------------------------------------------------

def ingest_leandojo_edges(db: duckdb.DuckDBPyConnection) -> int:
    """
    Parse all train/val/test.json splits and extract dependency edges from
    annotated_tactic[1] (list of {full_name, def_path, ...} premise references).
    """
    split_files = []
    for glob in SPLIT_GLOBS:
        split_files.extend(sorted(BENCHMARK_DIR.glob(glob)))

    if not split_files:
        print(f"[edges] No split JSON files found in {BENCHMARK_DIR}", file=sys.stderr)
        print("[edges] Run `python -m pipeline.fetch` to download and extract them.", file=sys.stderr)
        return 0

    print(f"[edges] Processing {len(split_files)} split file(s) for edges …")

    edges: set[tuple[str, str]] = set()
    # Track bare dst nodes not yet in declarations
    bare_nodes: dict[str, str | None] = {}  # full_name → def_path

    for split_path in split_files:
        print(f"  Loading {split_path.relative_to(LEANDOJO_DIR)} …", end=" ", flush=True)
        data: list[dict] = json.loads(split_path.read_text(encoding="utf-8"))
        print(f"{len(data):,} theorems")

        for entry in data:
            src = entry.get("full_name", "")
            if not src:
                continue

            # Ensure src is registered as a bare node (will be overwritten by corpus data)
            if src not in bare_nodes:
                bare_nodes[src] = entry.get("file_path")

            for tactic in entry.get("traced_tactics", []):
                ann = tactic.get("annotated_tactic", ["", []])
                if len(ann) < 2 or not ann[1]:
                    continue
                for premise in ann[1]:
                    dst = premise.get("full_name", "")
                    if dst and dst != src:
                        edges.add((src, dst))
                        if dst not in bare_nodes:
                            bare_nodes[dst] = premise.get("def_path")

    print(f"[edges] {len(edges):,} unique edges across {len(bare_nodes):,} nodes")

    # Ensure all nodes exist in declarations (bare insert, won't overwrite good data)
    node_rows = [
        {
            "name": name,
            "module": _module_from_filepath(fp) if fp else None,
            "module_parts": (
                _module_from_filepath(fp).split(".") if fp and _module_from_filepath(fp) else []
            ),
            "kind": "unknown",
            "type_sig": None,
            "in_leandojo": True,
            "in_mltypes": False,
        }
        for name, fp in bare_nodes.items()
    ]
    node_df = pl.DataFrame(
        node_rows,
        schema={
            "name": pl.String,
            "module": pl.String,
            "module_parts": pl.List(pl.String),
            "kind": pl.String,
            "type_sig": pl.String,
            "in_leandojo": pl.Boolean,
            "in_mltypes": pl.Boolean,
        },
    )
    db.register("bare_nodes", node_df.to_arrow())
    db.execute("""
        INSERT OR IGNORE INTO declarations
            (name, module, module_parts, kind, type_sig, in_leandojo, in_mltypes)
        SELECT name, module, module_parts, kind, type_sig, in_leandojo, in_mltypes
        FROM bare_nodes
    """)
    db.execute("UPDATE declarations SET in_leandojo = TRUE WHERE name IN (SELECT name FROM bare_nodes)")
    db.unregister("bare_nodes")

    # Insert edges
    edge_df = pl.DataFrame(
        [{"src": s, "dst": d} for s, d in edges],
        schema={"src": pl.String, "dst": pl.String},
    )
    db.register("leandojo_edges", edge_df.to_arrow())
    inserted_edges = db.execute("""
        INSERT OR IGNORE INTO dependencies (src, dst)
        SELECT src, dst FROM leandojo_edges
        WHERE EXISTS (SELECT 1 FROM declarations WHERE name = src)
          AND EXISTS (SELECT 1 FROM declarations WHERE name = dst)
    """).rowcount
    db.unregister("leandojo_edges")

    print(f"[edges] Inserted {inserted_edges:,} dependency edges")
    return inserted_edges


# ---------------------------------------------------------------------------
# Step 2b-extra: corpus.jsonl → lean_source table
# ---------------------------------------------------------------------------

def ingest_lean_source(db: duckdb.DuckDBPyConnection) -> int:
    """
    Populate lean_source(name, code, file_path, start_line, end_line)
    from corpus.jsonl. Each premise entry has full_name, code, start, end.
    """
    if not CORPUS_PATH.exists():
        print(f"[lean_source] corpus.jsonl not found at {CORPUS_PATH}", file=sys.stderr)
        return 0

    print(f"[lean_source] Reading corpus.jsonl for source code …")
    rows: list[dict] = []

    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in tqdm(f, desc="corpus.jsonl", unit="files"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            file_path = entry.get("path", "")
            for prem in entry.get("premises", []):
                name = prem.get("full_name", "")
                code = prem.get("code", "")
                start = prem.get("start", [None])
                end   = prem.get("end",   [None])
                if name and code:
                    rows.append({
                        "name":       name,
                        "code":       code,
                        "file_path":  file_path,
                        "start_line": start[0] if start else None,
                        "end_line":   end[0]   if end   else None,
                    })

    print(f"[lean_source] {len(rows):,} declarations with source code")

    df = pl.DataFrame(
        rows,
        schema={
            "name":       pl.String,
            "code":       pl.String,
            "file_path":  pl.String,
            "start_line": pl.Int32,
            "end_line":   pl.Int32,
        },
    )
    db.register("lean_source_batch", df.to_arrow())
    db.execute("INSERT OR IGNORE INTO lean_source SELECT * FROM lean_source_batch")
    db.unregister("lean_source_batch")

    stored = db.execute("SELECT COUNT(*) FROM lean_source").fetchone()[0]
    print(f"[lean_source] {stored:,} rows in lean_source table")
    return stored


# ---------------------------------------------------------------------------
# Step 2c: Split JSONs → tactics + tactic_deps tables
# ---------------------------------------------------------------------------

def ingest_tactic_data(db: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    """
    Populate the `tactics` and `tactic_deps` tables from all split JSON files.

    Each entry in a split JSON has:
      full_name, traced_tactics: [{tactic, annotated_tactic, state_before, state_after}]
    where annotated_tactic = [annotated_str, [{full_name, def_path, ...}, ...]]

    The two split families (random/, novel_premises/) overlap heavily — we use
    INSERT OR IGNORE so duplicates are silently skipped.
    """
    split_files = []
    for glob in SPLIT_GLOBS:
        split_files.extend(sorted(BENCHMARK_DIR.glob(glob)))

    if not split_files:
        print(f"[tactics] No split JSON files found in {BENCHMARK_DIR}", file=sys.stderr)
        return 0, 0

    print(f"[tactics] Ingesting tactic data from {len(split_files)} split file(s) …")

    total_tactics = 0
    total_deps = 0

    for split_path in split_files:
        print(f"  {split_path.relative_to(LEANDOJO_DIR)} …", end=" ", flush=True)
        data: list[dict] = json.loads(split_path.read_text(encoding="utf-8"))

        tactic_rows: list[dict] = []
        dep_rows: list[dict] = []
        seen_tactic_deps: set[tuple[str, int, str]] = set()

        for entry in data:
            theorem_name = entry.get("full_name", "")
            if not theorem_name:
                continue
            for idx, tactic in enumerate(entry.get("traced_tactics", [])):
                tactic_text = tactic.get("tactic", "")
                state_before = tactic.get("state_before", "")
                state_after = tactic.get("state_after", "")

                tactic_rows.append({
                    "theorem_name": theorem_name,
                    "tactic_idx": idx,
                    "tactic_text": tactic_text,
                    "state_before": state_before,
                    "state_after": state_after,
                })

                ann = tactic.get("annotated_tactic", ["", []])
                premises = ann[1] if len(ann) > 1 else []
                for prem in premises:
                    pname = prem.get("full_name", "")
                    if not pname:
                        continue
                    key = (theorem_name, idx, pname)
                    if key in seen_tactic_deps:
                        continue
                    seen_tactic_deps.add(key)
                    dep_rows.append({
                        "theorem_name": theorem_name,
                        "tactic_idx": idx,
                        "premise_name": pname,
                        "def_path": prem.get("def_path", ""),
                    })

        # Batch insert tactics
        if tactic_rows:
            t_df = pl.DataFrame(
                tactic_rows,
                schema={
                    "theorem_name": pl.String,
                    "tactic_idx": pl.Int32,
                    "tactic_text": pl.String,
                    "state_before": pl.String,
                    "state_after": pl.String,
                },
            )
            db.register("batch_tactics", t_df.to_arrow())
            n_t = db.execute(
                "INSERT OR IGNORE INTO tactics SELECT * FROM batch_tactics"
            ).rowcount
            db.unregister("batch_tactics")
            total_tactics += len(tactic_rows)

        # Batch insert tactic_deps
        if dep_rows:
            d_df = pl.DataFrame(
                dep_rows,
                schema={
                    "theorem_name": pl.String,
                    "tactic_idx": pl.Int32,
                    "premise_name": pl.String,
                    "def_path": pl.String,
                },
            )
            db.register("batch_deps", d_df.to_arrow())
            db.execute(
                "INSERT OR IGNORE INTO tactic_deps SELECT * FROM batch_deps"
            )
            db.unregister("batch_deps")
            total_deps += len(dep_rows)

        print(f"{len(tactic_rows):,} tactics, {len(dep_rows):,} deps")

    real_tactics = db.execute("SELECT COUNT(*) FROM tactics").fetchone()[0]
    real_deps = db.execute("SELECT COUNT(*) FROM tactic_deps").fetchone()[0]
    print(f"[tactics] Done — {real_tactics:,} tactic steps, {real_deps:,} tactic-level deps in DB")
    return real_tactics, real_deps


# ---------------------------------------------------------------------------
# Step 3: Post-processing
# ---------------------------------------------------------------------------

def postprocess(db: duckdb.DuckDBPyConnection) -> None:
    print("[postprocess] Filling missing module_parts …")
    db.execute("""
        UPDATE declarations
        SET module_parts = string_split(module, '.')
        WHERE module IS NOT NULL
          AND (module_parts IS NULL OR len(module_parts) = 0)
    """)

    total   = db.execute("SELECT COUNT(*) FROM declarations").fetchone()[0]
    both    = db.execute("SELECT COUNT(*) FROM declarations WHERE in_leandojo AND in_mltypes").fetchone()[0]
    only_ml = db.execute("SELECT COUNT(*) FROM declarations WHERE in_mltypes AND NOT in_leandojo").fetchone()[0]
    only_ld = db.execute("SELECT COUNT(*) FROM declarations WHERE in_leandojo AND NOT in_mltypes").fetchone()[0]
    edges   = db.execute("SELECT COUNT(*) FROM dependencies").fetchone()[0]
    tactics = db.execute("SELECT COUNT(*) FROM tactics").fetchone()[0]
    tdeps   = db.execute("SELECT COUNT(*) FROM tactic_deps").fetchone()[0]

    print("\n=== Ingest Summary ===")
    print(f"  declarations total : {total:>10,}")
    print(f"  in both datasets   : {both:>10,}")
    print(f"  only mathlib-types : {only_ml:>10,}")
    print(f"  only leandojo      : {only_ld:>10,}")
    print(f"  dependency edges   : {edges:>10,}")
    print(f"  tactic steps       : {tactics:>10,}")
    print(f"  tactic-level deps  : {tdeps:>10,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest raw data into DuckDB")
    parser.add_argument(
        "--only",
        choices=["mltypes", "leandojo"],
        help="Run only one ingestion step",
    )
    parser.add_argument("--db", default=str(DB_PATH), help="Path to DuckDB database file")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ingest] Opening DuckDB at {db_path}")
    db = duckdb.connect(str(db_path))
    db.execute(SCHEMA_SQL.read_text())

    if args.only != "leandojo":
        ingest_mathlib_types(db)

    if args.only != "mltypes":
        ingest_corpus_declarations(db)
        ingest_lean_source(db)
        ingest_leandojo_edges(db)
        ingest_tactic_data(db)

    postprocess(db)
    db.close()
    print(f"\n[ingest] Database ready: {db_path}")


if __name__ == "__main__":
    main()
