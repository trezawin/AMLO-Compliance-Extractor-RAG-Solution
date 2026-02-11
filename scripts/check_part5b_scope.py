# python scripts/check_part5b_scope.py --chunks data/processed/cap615.jsonl

import argparse
import json
import re
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that only the expected Part 5B sections are present in the chunks."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/cap615.jsonl"),
        help="Path to Cap.615 chunks JSONL",
    )
    args = parser.parse_args()

    seen = set()
    part5b_pattern = re.compile(r"^53ZR[A-Z]+$")
    with args.chunks.open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            section = chunk.get("section")
            block = chunk.get("top_level_block")
            if not section:
                continue

            # Prefer explicit block metadata when present.
            if block is not None:
                if block == "Part 5B" and section in ALLOWED_PART5B_SECTIONS:
                    seen.add(section)
                continue

            # Backward-compatible fallback for old chunk files:
            # only consider section IDs that look like 53ZR*.
            if part5b_pattern.match(str(section)) and section in ALLOWED_PART5B_SECTIONS:
                seen.add(section)

    unexpected = seen - ALLOWED_PART5B_SECTIONS
    missing = ALLOWED_PART5B_SECTIONS - seen

    print("Allowed Part 5B sections:", sorted(ALLOWED_PART5B_SECTIONS))
    print("Seen Part 5B sections   :", sorted(seen))
    print("Unexpected sections     :", sorted(unexpected) or "None")
    print("Missing sections        :", sorted(missing) or "None")


if __name__ == "__main__":
    main()
