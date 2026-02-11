import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

ALLOWED_PART5B_SECTIONS = {
    "53ZRD",
    "53ZRE",
    "53ZRF",
    "53ZRG",
    "53ZRH",
    "53ZRI",
    "53ZRJ",
    "53ZRK",
    "53ZRL",
}


def load_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check consistency between chunks, embeddings index, BM25 store, and scope metadata."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/cap615.jsonl"),
        help="Path to Cap.615 chunks JSONL",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/processed/embeddings.npy"),
        help="Path to embeddings .npy file",
    )
    parser.add_argument(
        "--bm25",
        type=Path,
        default=Path("data/processed/bm25.pkl"),
        help="Path to BM25 pickle file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any warning is found.",
    )
    args = parser.parse_args()

    warnings = []
    chunks = load_jsonl(args.chunks)
    chunks_count = len(chunks)
    embeddings = np.load(args.embeddings)
    meta_path = args.embeddings.with_suffix(".meta.jsonl")
    meta = load_jsonl(meta_path) if meta_path.exists() else []
    with args.bm25.open("rb") as f:
        payload = pickle.load(f)
    bm25_corpus_size = len(payload.get("corpus", []))
    bm25_chunks_size = len(payload.get("chunks", []))
    bm25_chunks = payload.get("chunks", [])

    print(f"Chunks JSONL count     : {chunks_count}")
    print(f"Embeddings rows        : {embeddings.shape[0]}")
    print(f"Embeddings meta rows   : {len(meta) if meta else 'MISSING'}")
    print(f"BM25 corpus size       : {bm25_corpus_size}")
    print(f"BM25 stored chunks size: {bm25_chunks_size}")

    if embeddings.shape[0] != chunks_count:
        warnings.append("embeddings row count does not match chunks count.")
    if meta and len(meta) != chunks_count:
        warnings.append("embeddings metadata row count does not match chunks count.")
    if bm25_chunks_size != chunks_count:
        warnings.append("BM25 stored chunks size does not match chunks count.")
    if bm25_corpus_size != bm25_chunks_size:
        warnings.append("BM25 corpus size does not match BM25 chunk count.")

    # Spot-check content alignment across files.
    if meta:
        if meta[0].get("text") != chunks[0].get("text"):
            warnings.append("first meta record text differs from chunks JSONL.")
        if meta[-1].get("text") != chunks[-1].get("text"):
            warnings.append("last meta record text differs from chunks JSONL.")
    if bm25_chunks:
        if bm25_chunks[0].get("text") != chunks[0].get("text"):
            warnings.append("first BM25 chunk text differs from chunks JSONL.")
        if bm25_chunks[-1].get("text") != chunks[-1].get("text"):
            warnings.append("last BM25 chunk text differs from chunks JSONL.")

    # Scope checks based on latest ingest schema.
    missing_block_meta = sum(1 for c in chunks if c.get("top_level_block") is None)
    if missing_block_meta:
        warnings.append(
            f"{missing_block_meta} chunks are missing top_level_block metadata "
            "(re-run ingest with latest code)."
        )

    part5b_seen = sorted(
        {
            c.get("section")
            for c in chunks
            if c.get("top_level_block") == "Part 5B" and c.get("section") in ALLOWED_PART5B_SECTIONS
        }
    )
    missing_part5b = sorted(ALLOWED_PART5B_SECTIONS - set(part5b_seen))
    sched2_count = sum(1 for c in chunks if c.get("top_level_block") == "Schedule 2")
    sched3b_count = sum(1 for c in chunks if c.get("top_level_block") == "Schedule 3B")

    print("\nScope checks:")
    print(f"Part 5B sections seen  : {part5b_seen}")
    print(f"Missing Part 5B        : {missing_part5b or 'None'}")
    print(f"Schedule 2 chunks      : {sched2_count}")
    print(f"Schedule 3B chunks     : {sched3b_count}")

    if missing_part5b:
        warnings.append(
            "missing required Part 5B sections: " + ", ".join(missing_part5b)
        )
    if sched2_count == 0:
        warnings.append("no chunks found for Schedule 2.")
    if sched3b_count == 0:
        warnings.append("no chunks found for Schedule 3B.")

    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"- {item}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("\nAll consistency checks passed.")


if __name__ == "__main__":
    main()
