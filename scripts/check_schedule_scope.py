import argparse
import json
import re
from collections import Counter
from pathlib import Path


ALLOWED_BLOCKS = {"Schedule 2", "Schedule 3B"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that Schedule 2 and Schedule 3B content is present in chunks."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/cap615.jsonl"),
        help="Path to Cap.615 chunks JSONL",
    )
    args = parser.parse_args()

    block_counts = Counter()
    section_counts = Counter()
    missing_block_metadata = 0

    chunks = []
    with args.chunks.open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunks.append(chunk)
            block = chunk.get("top_level_block")
            section = chunk.get("section")
            if block is None:
                missing_block_metadata += 1
                continue
            if block not in ALLOWED_BLOCKS:
                continue
            block_counts[block] += 1
            section_counts[(block, section)] += 1

    sched2_text = "\n".join(c.get("text", "") for c in chunks if c.get("top_level_block") == "Schedule 2")
    sched3b_text = "\n".join(c.get("text", "") for c in chunks if c.get("top_level_block") == "Schedule 3B")

    print("Block chunk counts:")
    for block in sorted(ALLOWED_BLOCKS):
        print(f"  {block}: {block_counts.get(block, 0)}")

    print("\nTop sections per block:")
    for block in sorted(ALLOWED_BLOCKS):
        print(f"  {block}:")
        rows = [(sec, n) for (b, sec), n in section_counts.items() if b == block]
        rows.sort(key=lambda x: x[1], reverse=True)
        if not rows:
            print("    (none)")
            continue
        for sec, n in rows[:12]:
            print(f"    {sec!r}: {n}")

    markers = [
        ("Schedule 2 marker: 1 Interpretation", bool(re.search(r"\b1\.?\s+Interpretation\b", sched2_text))),
        (
            "Schedule 2 marker: 2 Customer due diligence",
            bool(re.search(r"\b2\.?\s+Customer due diligence", sched2_text, flags=re.IGNORECASE)),
        ),
        (
            "Schedule 2 marker: 23 Financial institutions and DNFBPs",
            bool(re.search(r"\b23\.?\s+Financial institutions and DNFBPs", sched2_text, flags=re.IGNORECASE)),
        ),
        (
            "Schedule 3B marker: Operating a V A exchange",
            bool(re.search(r"Operating a V A exchange", sched3b_text, flags=re.IGNORECASE)),
        ),
        ("Schedule 3B marker: 1 Interpretation", bool(re.search(r"\b1\.?\s+Interpretation\b", sched3b_text))),
    ]
    print("\nMarker checks:")
    for label, ok in markers:
        print(f"  {label}: {'OK' if ok else 'MISSING'}")

    if missing_block_metadata:
        print(
            f"\nNote: {missing_block_metadata} chunk(s) had no top_level_block metadata. "
            "Re-run ingest with the latest code if this is unexpected."
        )


if __name__ == "__main__":
    main()
