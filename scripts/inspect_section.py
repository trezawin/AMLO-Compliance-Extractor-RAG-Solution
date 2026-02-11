
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect chunks for a given section ID (e.g. 53ZRE)."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/cap615.jsonl"),
        help="Path to Cap.615 chunks JSONL",
    )
    parser.add_argument(
        "--section",
        required=True,
        help="Section identifier to inspect (e.g. 53ZRE, 53ZRD, 'Schedule 2')",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=3,
        help="Maximum number of chunks to show",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=1000,
        help="Maximum characters of text to display per chunk",
    )
    args = parser.parse_args()

    count = 0
    with args.chunks.open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk.get("section") != args.section:
                continue
            count += 1
            if count > args.show:
                continue
            heading = chunk.get("heading")
            print("=" * 80)
            print(f"Section: {chunk.get('section')!r}")
            print(f"Heading: {heading!r}")
            text = chunk.get("text", "")
            if len(text) > args.max_chars:
                text = text[: args.max_chars] + "... [truncated]"
            print(text)

    print(f"\nTotal chunks for section {args.section!r}: {count}")


if __name__ == "__main__":
    main()

