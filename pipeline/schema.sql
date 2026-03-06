-- Mathlib Atlas DuckDB Schema

CREATE TABLE IF NOT EXISTS declarations (
    name         TEXT PRIMARY KEY,   -- e.g. 'Nat.add_comm'
    module       TEXT,               -- e.g. 'Mathlib.Data.Nat.Basic'
    module_parts TEXT[],             -- ['Mathlib','Data','Nat','Basic']
    kind         TEXT,               -- 'theorem'|'def'|'lemma'|'instance'|'axiom'|'unknown'
    type_sig     TEXT,               -- full type signature string
    in_leandojo  BOOL DEFAULT FALSE,
    in_mltypes   BOOL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS dependencies (
    src  TEXT REFERENCES declarations(name),  -- theorem that uses...
    dst  TEXT REFERENCES declarations(name),  -- ...this premise
    PRIMARY KEY (src, dst)
);

CREATE INDEX IF NOT EXISTS idx_declarations_module
    ON declarations(module);

CREATE INDEX IF NOT EXISTS idx_declarations_kind
    ON declarations(kind);

CREATE INDEX IF NOT EXISTS idx_dependencies_dst
    ON dependencies(dst);

CREATE INDEX IF NOT EXISTS idx_dependencies_src
    ON dependencies(src);

-- Per-tactic proof steps (only for tactic-mode theorems, ~50% of leandojo)
CREATE TABLE IF NOT EXISTS tactics (
    theorem_name  TEXT NOT NULL,  -- FK → declarations(name)
    tactic_idx    INT  NOT NULL,  -- 0-based step index within the proof
    tactic_text   TEXT,           -- raw tactic string, e.g. 'simp [Nat.add_comm]'
    state_before  TEXT,           -- proof state (hypotheses + goal) before this tactic
    state_after   TEXT,           -- proof state after ('no goals' when done)
    PRIMARY KEY (theorem_name, tactic_idx)
);

-- Which declarations each tactic step directly references
CREATE TABLE IF NOT EXISTS tactic_deps (
    theorem_name  TEXT NOT NULL,
    tactic_idx    INT  NOT NULL,
    premise_name  TEXT NOT NULL,  -- FK → declarations(name)
    def_path      TEXT,           -- source file of the premise
    PRIMARY KEY (theorem_name, tactic_idx, premise_name)
);

CREATE INDEX IF NOT EXISTS idx_tactics_theorem
    ON tactics(theorem_name);

CREATE INDEX IF NOT EXISTS idx_tactic_deps_premise
    ON tactic_deps(premise_name);

CREATE INDEX IF NOT EXISTS idx_tactic_deps_theorem
    ON tactic_deps(theorem_name);

-- Human-readable docstrings fetched from Mathlib GitHub source
CREATE TABLE IF NOT EXISTS docstrings (
    name      TEXT PRIMARY KEY,  -- FK → declarations(name)
    docstring TEXT NOT NULL
);

-- Raw Lean source code for each declaration (from corpus.jsonl)
CREATE TABLE IF NOT EXISTS lean_source (
    name       TEXT PRIMARY KEY,  -- FK → declarations(name)
    code       TEXT NOT NULL,     -- full declaration text (no docstring)
    file_path  TEXT,              -- e.g. 'Mathlib/LinearAlgebra/Matrix/Basic.lean'
    start_line INT,               -- 1-indexed line in file
    end_line   INT
);
