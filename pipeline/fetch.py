"""
fetch.py — Download raw datasets for Mathlib Metaanalysis.

Sources:
  1. mathlib-initiative/mathlib-types  (HuggingFace, Parquet)
  2. LeanDojo Benchmark 4              (Zenodo record 12740403, tar.gz)
"""

from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MLTYPES_DIR = RAW_DIR / "mathlib_types"
LEANDOJO_DIR = RAW_DIR / "leandojo"

# Zenodo record for LeanDojo Benchmark 4
ZENODO_RECORD_ID = "12740403"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


# ---------------------------------------------------------------------------
# HuggingFace: mathlib-types
# ---------------------------------------------------------------------------

def fetch_mathlib_types(force: bool = False) -> None:
    """Download mathlib-initiative/mathlib-types Parquet shards."""
    from huggingface_hub import snapshot_download

    if not force and MLTYPES_DIR.exists() and any(MLTYPES_DIR.glob("*.parquet")):
        print(f"[mathlib-types] Already downloaded at {MLTYPES_DIR}, skipping.")
        return

    MLTYPES_DIR.mkdir(parents=True, exist_ok=True)
    print("[mathlib-types] Downloading from HuggingFace …")
    snapshot_download(
        repo_id="mathlib-initiative/mathlib-types",
        repo_type="dataset",
        local_dir=str(MLTYPES_DIR),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    parquet_count = len(list(MLTYPES_DIR.rglob("*.parquet")))
    print(f"[mathlib-types] Done — {parquet_count} Parquet file(s) in {MLTYPES_DIR}")


# ---------------------------------------------------------------------------
# Zenodo: LeanDojo Benchmark 4
# ---------------------------------------------------------------------------

def _zenodo_get_files() -> list[dict]:
    """Fetch the file list for a Zenodo record."""
    resp = requests.get(ZENODO_API_URL, timeout=30)
    resp.raise_for_status()
    record = resp.json()
    return record["files"]


def _download_with_progress(url: str, dest: Path) -> None:
    """Stream-download url → dest with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        with open(dest, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))


_EXTRACT_TARGETS = [
    "corpus.jsonl",
    "random/train.json",
    "random/val.json",
    "random/test.json",
    "novel_premises/train.json",
    "novel_premises/val.json",
    "novel_premises/test.json",
]


def fetch_leandojo(force: bool = False) -> None:
    """Download and extract the LeanDojo Benchmark 4 archive from Zenodo."""
    benchmark_dir = LEANDOJO_DIR / "leandojo_benchmark_4"
    all_present = (LEANDOJO_DIR / "corpus.jsonl").exists() and all(
        (benchmark_dir / t).exists() for t in _EXTRACT_TARGETS[1:]
    )
    if not force and all_present:
        print(f"[leandojo] All files already present in {LEANDOJO_DIR}, skipping.")
        return

    LEANDOJO_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve file list from Zenodo API
    print(f"[leandojo] Querying Zenodo record {ZENODO_RECORD_ID} …")
    try:
        files = _zenodo_get_files()
    except Exception as exc:
        print(f"[leandojo] ERROR: Could not fetch Zenodo metadata: {exc}", file=sys.stderr)
        print(
            "[leandojo] Tip: Download manually from "
            f"https://zenodo.org/records/{ZENODO_RECORD_ID} "
            f"and extract corpus.jsonl to {LEANDOJO_DIR}/",
            file=sys.stderr,
        )
        return

    # Pick the main archive (prefer the .tar.gz containing corpus.jsonl)
    archive_file = next(
        (f for f in files if f["key"].endswith(".tar.gz")),
        None,
    )
    if archive_file is None:
        # Fallback: look for corpus.jsonl directly
        corpus_file = next((f for f in files if "corpus" in f["key"]), None)
        if corpus_file is None:
            print(f"[leandojo] ERROR: Cannot find corpus file in Zenodo record.", file=sys.stderr)
            print(f"  Available files: {[f['key'] for f in files]}", file=sys.stderr)
            return
        dest = LEANDOJO_DIR / corpus_file["key"]
        _download_with_progress(corpus_file["links"]["self"], dest)
        print(f"[leandojo] Downloaded {dest}")
        return

    archive_dest = LEANDOJO_DIR / archive_file["key"]
    if not force and archive_dest.exists():
        print(f"[leandojo] Archive already downloaded at {archive_dest}")
    else:
        print(f"[leandojo] Downloading {archive_file['key']} ({archive_file['size'] / 1e6:.0f} MB) …")
        _download_with_progress(archive_file["links"]["self"], archive_dest)

    # Extract: corpus.jsonl (flattened) + split JSONs (keep subdir structure)
    print(f"[leandojo] Extracting {archive_dest} …")
    with tarfile.open(archive_dest, "r:gz") as tf:
        all_members = tf.getmembers()
        # corpus.jsonl → flatten to LEANDOJO_DIR/corpus.jsonl
        for m in all_members:
            if m.name.endswith("corpus.jsonl"):
                m.name = "corpus.jsonl"
                tf.extract(m, path=LEANDOJO_DIR, filter="data")
                print(f"[leandojo] Extracted corpus.jsonl")
        # Split JSONs → keep as leandojo_benchmark_4/{random,novel_premises}/
        target_suffixes = [
            "random/train.json", "random/val.json", "random/test.json",
            "novel_premises/train.json", "novel_premises/val.json", "novel_premises/test.json",
        ]
        for m in all_members:
            for suffix in target_suffixes:
                if m.name.endswith(suffix):
                    tf.extract(m, path=LEANDOJO_DIR, filter="data")
                    print(f"[leandojo] Extracted {m.name}")
                    break

    if corpus_path.exists():
        size_mb = corpus_path.stat().st_size / 1e6
        print(f"[leandojo] corpus.jsonl ready ({size_mb:.1f} MB)")
    else:
        print(
            f"[leandojo] Warning: corpus.jsonl not found after extraction. "
            f"Check {LEANDOJO_DIR}/ for the actual file.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Mathlib Metaanalysis raw datasets")
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    parser.add_argument(
        "--only",
        choices=["mltypes", "leandojo"],
        help="Download only one dataset",
    )
    args = parser.parse_args()

    if args.only != "leandojo":
        fetch_mathlib_types(force=args.force)
    if args.only != "mltypes":
        fetch_leandojo(force=args.force)

    print("\nAll done.")


if __name__ == "__main__":
    main()
