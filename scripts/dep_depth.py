#!/usr/bin/env python3
"""
Show how deep the dependency tree goes for a given Lean 4 theorem.

For theorem T, the dependency tree is: T → (what T uses) → (what those use) → ...
We compute:
  - BFS levels (min distance from T to each transitive dependency)
  - Longest chain (max distance, computed via topological DP on the DAG)
  - An example of the longest chain, printed as a path

Usage:
    python3 scripts/dep_depth.py [theorem_name ...]

If no names given, runs a built-in set of examples spanning different complexity levels.
"""

import sys
from pathlib import Path
from collections import defaultdict, deque

import duckdb

ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "mathlib.db"

# ── Defaults (each was chosen to have a non-trivial but readable tree) ────────
DEFAULTS = [
    # Very shallow — many fundamental results are proved by omega/simp/decide
    # with no recorded premises, so chains terminate quickly.
    "Nat.factors_lemma",                         # 3 direct deps, depth 2
    "Finset.prod_range_succ",                    # depth 8, 24 transitive deps
    # Moderate depth — non-trivial proof chains through number theory
    "padicNorm.dvd_iff_norm_le",                 # p-adic norm lemma
    "PadicInt.norm_int_lt_one_iff_dvd",          # depth 12, 195 transitive deps
    # Wide but shallow — Fermat n=4 uses many lemmas but chain stays short
    "Fermat42.not_minimal",                      # depth 7, 194 transitive deps
]

# ─────────────────────────────────────────────────────────────────────────────

def load_edges(db: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """Return adjacency list adj[src] = [dst, ...]."""
    rows = db.execute("SELECT src, dst FROM dependencies").fetchall()
    adj: dict[str, list[str]] = defaultdict(list)
    for src, dst in rows:
        adj[src].append(dst)
    return adj


def reachable_bfs(adj: dict[str, list[str]], start: str) -> dict[str, int]:
    """BFS from *start* following outgoing edges. Returns {node: min_distance}."""
    dist: dict[str, int] = {start: 0}
    q: deque[str] = deque([start])
    while q:
        node = q.popleft()
        for nb in adj.get(node, []):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                q.append(nb)
    return dist


def longest_path_in_subgraph(
    adj: dict[str, list[str]],
    nodes: set[str],
    start: str,
) -> tuple[int, list[str]]:
    """
    Longest path from *start* in the subgraph induced by *nodes*.

    Uses Kahn's topological sort + DP (works correctly on DAGs;
    cycles are broken by ignoring back-edges in the topo pass).

    Returns (length, path) where path is the list of node names.
    """
    # Build induced in-degrees
    in_deg: dict[str, int] = {n: 0 for n in nodes}
    for n in nodes:
        for nb in adj.get(n, []):
            if nb in nodes:
                in_deg[nb] = in_deg.get(nb, 0) + 1

    # DP tables: best distance from *start*, predecessor for path reconstruction
    dp:   dict[str, int]        = {n: -1   for n in nodes}
    prev: dict[str, str | None] = {n: None for n in nodes}
    dp[start] = 0

    # Kahn's queue (starts from in-degree-0 nodes in the induced subgraph)
    queue: deque[str] = deque(n for n in nodes if in_deg[n] == 0)

    while queue:
        node = queue.popleft()
        for nb in adj.get(node, []):
            if nb not in nodes:
                continue
            # Relax edge
            if dp[node] >= 0 and dp[node] + 1 > dp[nb]:
                dp[nb]   = dp[node] + 1
                prev[nb] = node
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)

    # Find the farthest reachable node from start
    end = max((n for n in nodes if dp[n] >= 0), key=lambda n: dp[n], default=start)

    # Trace back the path
    path: list[str] = []
    cur: str | None = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    return dp[end], path


def describe(db: duckdb.DuckDBPyConnection, name: str) -> str:
    """Short one-line description of a declaration."""
    row = db.execute(
        "SELECT kind, module FROM declarations WHERE name = ?", [name]
    ).fetchone()
    if row is None:
        return "(unknown)"
    kind, module = row
    return f"{kind}  [{module}]"


def analyze(
    adj: dict[str, list[str]],
    db:  duckdb.DuckDBPyConnection,
    name: str,
) -> None:
    # Verify it exists
    if db.execute("SELECT 1 FROM declarations WHERE name = ?", [name]).fetchone() is None:
        print(f"\n✗  '{name}' not found in declarations table.\n")
        return

    dist = reachable_bfs(adj, name)
    reachable_nodes = set(dist.keys())
    total_deps       = len(reachable_nodes) - 1  # exclude the theorem itself

    if total_deps == 0:
        print(f"\n{'━'*70}")
        print(f"  {name}")
        print(f"  {describe(db, name)}")
        print(f"{'━'*70}")
        print(f"  No tracked dependencies (proved by tactic with no recorded premises,")
        print(f"  e.g., omega / decide / simp-closed, or a primitive axiom).")
        print()
        return

    max_bfs_depth = max(dist.values())

    # Longest-path DP
    longest_len, longest_path = longest_path_in_subgraph(adj, reachable_nodes, name)

    # Count how many nodes live at each BFS level
    level_count: dict[int, int] = defaultdict(int)
    for d in dist.values():
        level_count[d] += 1

    # Print
    print(f"\n{'━'*70}")
    print(f"  {name}")
    print(f"  {describe(db, name)}")
    print(f"{'━'*70}")
    print(f"  Transitive dependencies : {total_deps:,}")
    print(f"  Max BFS depth           : {max_bfs_depth}  "
          f"(shortest path from here to any dep at that level)")
    print(f"  Longest chain           : {longest_len}  "
          f"(longest sequence  A → B → C → … in the DAG)")
    print()

    # BFS-level histogram
    print("  Deps by BFS level (min hops from theorem):")
    for level in range(1, max_bfs_depth + 1):
        n = level_count.get(level, 0)
        bar = "█" * min(n, 40) + ("+" if n > 40 else "")
        print(f"    level {level:2d}: {n:5,}  {bar}")
    print()

    # Longest chain
    print(f"  Longest dependency chain ({longest_len + 1} steps):")
    for i, node in enumerate(longest_path):
        prefix = "  " + ("  → " if i > 0 else "    ")
        # Indent deeper levels, cap at 8
        indent = "     " * min(i, 8)
        arrow  = "→ "  if i > 0 else "  "
        # Truncate very long names
        disp = node if len(node) <= 58 else node[:55] + "…"
        kind_tag = ""
        row = db.execute(
            "SELECT kind FROM declarations WHERE name = ?", [node]
        ).fetchone()
        if row:
            kind_tag = f"  [{row[0]}]"
        print(f"    {'  ' * min(i, 10)}{'→ ' if i > 0 else ''}{disp}{kind_tag}")
    print()


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    names = sys.argv[1:] if len(sys.argv) > 1 else DEFAULTS

    db  = duckdb.connect(str(DB_PATH), read_only=True)

    print("Loading all dependency edges …", end=" ", flush=True)
    adj = load_edges(db)
    print(f"{sum(len(v) for v in adj.values()):,} edges loaded.")

    for name in names:
        analyze(adj, db, name)

    db.close()


if __name__ == "__main__":
    main()
