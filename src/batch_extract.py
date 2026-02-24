import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

from .config import paths
from .pipeline import call_llm, parse_extraction_response
from .prompts import build_prompt

console = Console()
VALID_CONTEXT_TYPES = {
    "DEFINITION",
    "OBLIGATION_TEXT",
    "GUIDANCE",
    "CONSEQUENCE",
    "SCOPE_NOTE",
    "MAPPING_NOTE",
}
VALID_COMPLIANCE_CATEGORIES = {
    "DUE_DILIGENCE",
    "ELIGIBILITY",
    "CONTROL",
    "TRANSFER_COMPLIANCE",
    "TRANSACTION_LIMIT",
}


def load_chunks(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def group_by_section(chunks: List[Dict], max_tokens: int) -> List[Dict]:
    """Combine contiguous chunks from the same section into contexts capped by token count."""
    grouped = []
    buffer: List[Dict] = []
    buffer_tokens = 0

    def flush():
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        section = buffer[0].get("section")
        heading = buffer[0].get("heading")
        top_level_block = buffer[0].get("top_level_block")
        top_level_kind = buffer[0].get("top_level_kind")
        texts = [c["text"] for c in buffer]
        context = "\n\n---\n\n".join(texts)
        grouped.append(
            {
                "section": section,
                "heading": heading,
                "top_level_block": top_level_block,
                "top_level_kind": top_level_kind,
                "context": context,
            }
        )
        buffer = []
        buffer_tokens = 0

    for chunk in chunks:
        tokens = chunk["tokens"]
        if not buffer:
            buffer = [chunk]
            buffer_tokens = tokens
            continue
        same_section = chunk.get("section") == buffer[-1].get("section")
        if same_section and buffer_tokens + tokens <= max_tokens:
            buffer.append(chunk)
            buffer_tokens += tokens
        else:
            flush()
            buffer = [chunk]
            buffer_tokens = tokens
    flush()
    return grouped


def build_human_ref(
    section: Optional[str],
    heading: Optional[str],
    top_level_block: Optional[str],
    top_level_kind: Optional[str],
) -> str:
    section_text = f"Section {section}" if section else "Section UNSPECIFIED"
    if top_level_kind in {"Part", "Schedule"} and top_level_block:
        base = f"{top_level_block}, {section_text}"
    else:
        base = section_text
    if heading:
        return f"{base} - {heading}"
    return base


def normalize_rule_ref(candidate: Optional[str], fallback_ref: str) -> str:
    if not candidate:
        return fallback_ref
    text = str(candidate).strip()
    if not text:
        return fallback_ref
    # Reject ambiguous refs like "12(2)(a)" without Part/Schedule/Section context.
    has_context = bool(re.search(r"\b(Part|Schedule|Section)\b", text, flags=re.IGNORECASE))
    bare_subref = bool(re.fullmatch(r"\d+[A-Za-z]?(?:\([^)]+\))*", text))
    if bare_subref and not has_context:
        return f"{fallback_ref}, subsection {text}"
    if not has_context:
        return f"{fallback_ref} - {text}"
    return text


def normalize_context_type(candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    value = str(candidate).strip().upper()
    return value if value in VALID_CONTEXT_TYPES else None


def normalize_compliance_category(candidate: Optional[str]) -> str:
    if not candidate:
        return "CONTROL"
    value = str(candidate).strip().upper()
    return value if value in VALID_COMPLIANCE_CATEGORIES else "CONTROL"


def normalize_check_method(candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    value = str(candidate).strip().lower()
    alias = {
        "deterministic": "deterministic",
        "binary-check": "deterministic",
        "observational": "observational",
        "observation": "observational",
        "evidence": "observational",
        "evidence-based": "observational",
    }
    return alias.get(value, value if value in {"deterministic", "observational"} else None)


def normalize_pass_criteria(candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    value = str(candidate).strip().lower()
    alias = {
        "binary": "binary",
        "pass/fail": "binary",
        "pass-fail": "binary",
        "evidence-based": "evidence-based",
        "evidence based": "evidence-based",
        "evidence": "evidence-based",
    }
    return alias.get(value, value if value in {"binary", "evidence-based"} else None)


def normalize_obligation_nature(candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    value = str(candidate).strip().lower()
    alias = {
        "prohibition": "prohibition",
        "must-not-have": "prohibition",
        "requirement": "requirement",
        "must-have": "requirement",
        "capability": "capability",
    }
    return alias.get(value, value if value in {"prohibition", "requirement", "capability"} else None)


def main():
    parser = argparse.ArgumentParser(description="Extract obligations across all sections and aggregate JSON.")
    parser.add_argument("--chunks", type=Path, default=paths.default_chunks, help="Path to JSONL chunks")
    parser.add_argument("--out", type=Path, default=paths.processed_dir / "all_rules.json", help="Output JSON file")
    parser.add_argument("--max_context_tokens", type=int, default=1500, help="Max tokens per grouped context")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick runs")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many grouped contexts before processing")
    parser.add_argument("--count", type=int, default=None, help="Process only this many grouped contexts after offset")
    parser.add_argument("--dry_run", action="store_true", help="Skip LLM calls; emit prompts only")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    grouped = group_by_section(chunks, max_tokens=args.max_context_tokens)
    if args.limit:
        grouped = grouped[: args.limit]
    if args.offset:
        grouped = grouped[args.offset :]
    if args.count is not None:
        grouped = grouped[: args.count]
    console.print(f"[cyan]Processing {len(grouped)} grouped contexts...")

    rules_out: List[Dict] = []
    contexts_out: List[Dict] = []
    rule_counter = 0
    context_counter = 0

    for idx, item in enumerate(grouped, 1):
        context = item["context"]
        normalized_context = " ".join(context.split())
        section_label = item.get("section") or f"group-{idx}"
        query = "Extract enforceable obligations in this section."
        prompt = build_prompt(context=context, query=query)
        if args.dry_run:
            console.print(f"[yellow]Dry run for section {section_label}; prompt length={len(prompt)}")
            continue
        resp = call_llm(prompt)
        parsed = parse_extraction_response(resp)
        section = item.get("section")
        heading = item.get("heading")
        top_level_block = item.get("top_level_block")
        top_level_kind = item.get("top_level_kind")
        ref = build_human_ref(section, heading, top_level_block, top_level_kind)

        # Normalize and enrich rules.
        for rule in parsed.get("rules", []):
            check_method = normalize_check_method(rule.get("check_method"))
            pass_criteria = normalize_pass_criteria(rule.get("pass_criteria"))
            obligation_nature = normalize_obligation_nature(rule.get("obligation_nature"))
            original_text = (rule.get("original_text") or "").strip()
            # Enforce the minimal gating contract: keep only rules that
            # have a claim + check_method + pass_criteria.
            if not rule.get("rule_objective"):
                continue
            if check_method not in {"deterministic", "observational"}:
                continue
            if pass_criteria not in {"binary", "evidence-based"}:
                continue
            if not original_text:
                continue
            # Drop hallucinated or paraphrased clauses: require verbatim substring of the context.
            normalized_original = " ".join(original_text.split())
            if original_text not in context and normalized_original not in normalized_context:
                continue
            rule_counter += 1
            normalized = {
                "rule_id": f"R-{rule_counter}",
                "rule_ref": normalize_rule_ref(rule.get("rule_ref"), ref),
                "compliance_category": normalize_compliance_category(rule.get("compliance_category")),
                "rule_objective": rule.get("rule_objective"),
                "original_text": original_text,
                "obligation_nature": obligation_nature,
                "check_method": check_method,
                "pass_criteria": pass_criteria,
                # enforcement_interface intentionally omitted for now.
            }
            rules_out.append(normalized)

        # Normalize and enrich contexts.
        for ctx in parsed.get("contexts", []):
            context_type = normalize_context_type(ctx.get("context_type"))
            text = (ctx.get("text") or "").strip()
            if not context_type or not text:
                continue
            normalized_text = " ".join(text.split())
            # Drop hallucinated or paraphrased context snippets: require verbatim substring of the context.
            if text not in context and normalized_text not in normalized_context:
                continue
            context_counter += 1
            normalized_ctx = {
                "context_id": f"C-{context_counter}",
                "source_ref": normalize_rule_ref(ctx.get("source_ref"), ref),
                "context_type": context_type,
                "text": text,
                "notes": (ctx.get("notes") or "").strip() or None,
            }
            contexts_out.append(normalized_ctx)

        if idx % 10 == 0:
            console.print(
                f"[green]Processed {idx}/{len(grouped)} contexts; "
                f"total rules: {len(rules_out)}, contexts: {len(contexts_out)}"
            )

    if args.dry_run:
        console.print("[yellow]Dry run complete; no output file written.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rules": rules_out, "contexts": contexts_out}
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    console.print(f"[bold green]Wrote {len(rules_out)} rules and {len(contexts_out)} contexts to {args.out}")


if __name__ == "__main__":
    main()
