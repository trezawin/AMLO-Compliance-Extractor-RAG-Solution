import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_extraction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run extraction for a query and save only the top 5 rules."
    )
    parser.add_argument("--query", required=True, help="Extraction query")
    parser.add_argument("--top_k", type=int, default=8, help="Retriever top_k")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/top5_rules.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip LLM call; save prompt/retrieval only.",
    )
    args = parser.parse_args()

    payload = run_extraction(query=args.query, top_k=args.top_k, dry_run=args.dry_run)
    rules = payload.get("rules", [])
    top5 = rules[:5]

    output = {
        "query": payload.get("query"),
        "top_k": payload.get("top_k"),
        "rules_total": len(rules),
        "rules_top_5": top5,
        "contexts": payload.get("contexts", []),
        "retrieved": payload.get("retrieved", []),
    }
    if args.dry_run:
        output["prompt"] = payload.get("prompt")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Wrote top 5 rules ({len(top5)}) to {args.out}")


if __name__ == "__main__":
    main()
