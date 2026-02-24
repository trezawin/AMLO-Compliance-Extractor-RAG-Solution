import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def load_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_ref(chunk: Dict) -> str:
    top = chunk.get("top_level_block") or chunk.get("part") or chunk.get("schedule") or "Unknown"
    section = chunk.get("section") or chunk.get("heading")
    if section:
        return f"Cap.615 {top} {section}".strip()
    return f"Cap.615 {top}".strip()


def sample_chunks(chunks: List[Dict], *, seed: int, n: int) -> List[Dict]:
    if n <= 0:
        return []
    rng = random.Random(seed)
    if len(chunks) <= n:
        return list(chunks)
    return rng.sample(chunks, n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a gold-set skeleton JSONL for human labeling (RULE vs CONTEXT) "
            "from ingested Cap.615 chunks."
        )
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/cap615.jsonl"),
        help="Path to chunks JSONL",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/gold_skeleton.jsonl"),
        help="Output gold skeleton JSONL",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed")
    parser.add_argument(
        "--schedule2",
        type=int,
        default=60,
        help="How many Schedule 2 chunks to include (0 disables)",
    )
    parser.add_argument(
        "--schedule3b",
        type=int,
        default=20,
        help="How many Schedule 3B chunks to include (0 disables)",
    )
    parser.add_argument(
        "--part5b",
        type=int,
        default=9,
        help="How many Part 5B chunks to include (0 disables). Use 9 to include all 53ZRD–53ZRL.",
    )
    args = parser.parse_args()

    chunks = load_jsonl(args.chunks)
    sched2 = [c for c in chunks if c.get("top_level_block") == "Schedule 2"]
    sched3b = [c for c in chunks if c.get("top_level_block") == "Schedule 3B"]
    part5b = [c for c in chunks if c.get("top_level_block") == "Part 5B"]

    picked: List[Dict] = []
    picked.extend(sample_chunks(sched2, seed=args.seed + 2, n=args.schedule2))
    picked.extend(sample_chunks(sched3b, seed=args.seed + 3, n=args.schedule3b))
    picked.extend(sample_chunks(part5b, seed=args.seed + 5, n=args.part5b))

    # Stable ordering for review: group by source, then by id where present.
    def sort_key(c: Dict) -> str:
        return f"{c.get('top_level_block','')}-{c.get('section','')}-{c.get('id','')}"

    picked = sorted(picked, key=sort_key)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(picked, start=1):
            row = {
                "gold_id": f"gold_{i:04d}",
                # Human fills:
                "label": None,  # RULE | CONTEXT
                "source_ref": chunk_ref(chunk),
                "verbatim_text": chunk.get("text", ""),
                # If label=RULE, fill:
                "rule_objective": None,
                "compliance_category": None,
                "obligation_nature": None,
                "check_method": None,
                "pass_criteria": None,
                # If label=CONTEXT, fill:
                "context_type": None,
                "notes": None,
                # Provenance helpers (do not edit):
                "_chunk_id": chunk.get("id"),
                "_top_level_block": chunk.get("top_level_block"),
                "_section": chunk.get("section"),
                "_heading": chunk.get("heading"),
                "_source": chunk.get("source"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(picked)} gold skeleton rows to {args.out}")
    print(
        "Next: label each row with label=RULE or label=CONTEXT and fill the relevant fields."
    )


if __name__ == "__main__":
    main()

