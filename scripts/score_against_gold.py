import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ALLOWED_LABELS = {"RULE", "CONTEXT"}
ALLOWED_CONTEXT_TYPES = {
    "DEFINITION",
    "OBLIGATION_TEXT",
    "GUIDANCE",
    "CONSEQUENCE",
    "SCOPE_NOTE",
    "MAPPING_NOTE",
}
ALLOWED_OBLIGATION_NATURE = {"prohibition", "requirement", "capability"}
ALLOWED_CHECK_METHOD = {"deterministic", "observational"}
ALLOWED_PASS_CRITERIA = {"binary", "evidence-based"}
ALLOWED_COMPLIANCE_CATEGORIES = {
    "DUE_DILIGENCE",
    "ELIGIBILITY",
    "CONTROL",
    "TRANSFER_COMPLIANCE",
    "TRANSACTION_LIMIT",
}


def load_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pred(path: Path) -> Dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    # Accept both "all_rules.json" format and "top5_rules*.json" format.
    if "rules" in obj and "contexts" in obj:
        return obj
    if "rules_top_5" in obj:
        return {"rules": obj.get("rules_top_5", []), "contexts": obj.get("contexts", [])}
    raise SystemExit("Unrecognized prediction JSON format.")


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def norm_ref(ref: Optional[str]) -> str:
    if not ref:
        return ""
    ref = ref.lower()
    ref = re.sub(r"\s+", " ", ref).strip()
    # drop most punctuation but keep parentheses markers
    ref = re.sub(r"[^a-z0-9() .-]+", "", ref)
    return ref


def looks_like_citation(ref: str) -> bool:
    r = ref.lower()
    # Heuristic: must mention section or schedule, and some numbering/letters.
    has_anchor = ("sch" in r) or ("schedule" in r) or (" s." in r) or ("section" in r)
    has_digits = any(ch.isdigit() for ch in r)
    return bool(ref) and has_anchor and has_digits


def build_corpus(chunks_path: Path) -> str:
    chunks = load_jsonl(chunks_path)
    texts = [norm_ws(c.get("text", "")) for c in chunks]
    return "\n<<<CHUNK>>>\n".join(t for t in texts if t)


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def match_items(
    gold: Dict, candidates: List[Dict], *, ref_key: str, text_key: str
) -> bool:
    gold_ref = norm_ref(gold.get("source_ref"))
    gold_text = norm_ws(gold.get("verbatim_text", ""))
    if not gold_ref and not gold_text:
        return False

    for cand in candidates:
        cand_ref = norm_ref(cand.get(ref_key))
        cand_text = norm_ws(cand.get(text_key, ""))
        if gold_ref and cand_ref and gold_ref == cand_ref:
            return True
        if gold_text and cand_text:
            if gold_text in cand_text or cand_text in gold_text:
                return True
    return False


def validate_pred_rows(pred: Dict, corpus: Optional[str]) -> List[str]:
    warnings: List[str] = []
    rules = pred.get("rules", [])
    contexts = pred.get("contexts", [])

    bad_rule_fields = 0
    bad_ctx_fields = 0
    bad_citations = 0
    bad_verbatim = 0
    bad_categories = 0

    for r in rules:
        if not r.get("rule_ref") or not r.get("original_text"):
            bad_rule_fields += 1
        if r.get("obligation_nature") and r["obligation_nature"] not in ALLOWED_OBLIGATION_NATURE:
            bad_rule_fields += 1
        if r.get("check_method") and r["check_method"] not in ALLOWED_CHECK_METHOD:
            bad_rule_fields += 1
        if r.get("pass_criteria") and r["pass_criteria"] not in ALLOWED_PASS_CRITERIA:
            bad_rule_fields += 1
        if r.get("compliance_category") and r["compliance_category"] not in ALLOWED_COMPLIANCE_CATEGORIES:
            bad_categories += 1
        if not looks_like_citation(r.get("rule_ref", "")):
            bad_citations += 1
        if corpus is not None:
            if norm_ws(r.get("original_text", "")) and norm_ws(r.get("original_text", "")) not in corpus:
                bad_verbatim += 1

    for c in contexts:
        if not c.get("source_ref") or not c.get("text"):
            bad_ctx_fields += 1
        if c.get("context_type") and c["context_type"] not in ALLOWED_CONTEXT_TYPES:
            bad_ctx_fields += 1
        if not looks_like_citation(c.get("source_ref", "")):
            bad_citations += 1
        if corpus is not None:
            if norm_ws(c.get("text", "")) and norm_ws(c.get("text", "")) not in corpus:
                bad_verbatim += 1

    if bad_rule_fields:
        warnings.append(f"{bad_rule_fields} rule rows have missing/invalid schema fields.")
    if bad_ctx_fields:
        warnings.append(f"{bad_ctx_fields} context rows have missing/invalid schema fields.")
    if bad_categories:
        warnings.append(f"{bad_categories} rule rows have invalid compliance_category.")
    if bad_citations:
        warnings.append(f"{bad_citations} rows have weak/invalid-looking citations.")
    if corpus is not None and bad_verbatim:
        warnings.append(
            f"{bad_verbatim} rows have text that does not appear verbatim in chunks corpus."
        )
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score extracted rules/contexts JSON against a labeled gold JSONL."
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("eval/gold_adjudicated.jsonl"),
        help="Gold JSONL (labeled)",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=Path("data/processed/all_rules.json"),
        help="Prediction JSON (rules+contexts)",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/cap615.jsonl"),
        help="Chunks JSONL (for verbatim validation)",
    )
    parser.add_argument(
        "--report_out",
        type=Path,
        default=None,
        help="Optional path to write JSON report",
    )
    args = parser.parse_args()

    gold_all = load_jsonl(args.gold)
    gold = [g for g in gold_all if (g.get("label") in ALLOWED_LABELS)]
    if not gold:
        raise SystemExit("Gold file has no labeled rows (label must be RULE or CONTEXT).")

    pred = load_pred(args.pred)
    rules_pred = pred.get("rules", [])
    ctx_pred = pred.get("contexts", [])

    corpus = build_corpus(args.chunks) if args.chunks.exists() else None
    warnings = validate_pred_rows(pred, corpus)

    gold_rules = [g for g in gold if g.get("label") == "RULE"]
    gold_ctx = [g for g in gold if g.get("label") == "CONTEXT"]

    # Recall-style scoring: for each gold item, is there a matching predicted item?
    rule_score = Score()
    ctx_score = Score()

    for g in gold_rules:
        if match_items(g, rules_pred, ref_key="rule_ref", text_key="original_text"):
            rule_score.tp += 1
        else:
            rule_score.fn += 1

    for g in gold_ctx:
        if match_items(g, ctx_pred, ref_key="source_ref", text_key="text"):
            ctx_score.tp += 1
        else:
            ctx_score.fn += 1

    # Precision-style scoring: for each predicted item, does it match any gold item of same label?
    for r in rules_pred:
        pseudo_gold = {
            "source_ref": r.get("rule_ref"),
            "verbatim_text": r.get("original_text"),
        }
        if match_items(pseudo_gold, gold_rules, ref_key="source_ref", text_key="verbatim_text"):
            # already counted as TP for recall; for precision bookkeeping count FP separately
            pass
        else:
            rule_score.fp += 1

    for c in ctx_pred:
        pseudo_gold = {"source_ref": c.get("source_ref"), "verbatim_text": c.get("text")}
        if match_items(pseudo_gold, gold_ctx, ref_key="source_ref", text_key="verbatim_text"):
            pass
        else:
            ctx_score.fp += 1

    total_gold = len(gold_rules) + len(gold_ctx)
    total_pred = len(rules_pred) + len(ctx_pred)

    report = {
        "gold_total_labeled": total_gold,
        "gold_rules": len(gold_rules),
        "gold_contexts": len(gold_ctx),
        "pred_total": total_pred,
        "pred_rules": len(rules_pred),
        "pred_contexts": len(ctx_pred),
        "rules": {
            "tp": rule_score.tp,
            "fp": rule_score.fp,
            "fn": rule_score.fn,
            "precision": rule_score.precision,
            "recall": rule_score.recall,
            "f1": rule_score.f1,
        },
        "contexts": {
            "tp": ctx_score.tp,
            "fp": ctx_score.fp,
            "fn": ctx_score.fn,
            "precision": ctx_score.precision,
            "recall": ctx_score.recall,
            "f1": ctx_score.f1,
        },
        "warnings": warnings,
    }

    print("Gold labeled rows:", total_gold, f"(rules={len(gold_rules)}, contexts={len(gold_ctx)})")
    print("Pred rows        :", total_pred, f"(rules={len(rules_pred)}, contexts={len(ctx_pred)})")
    print("\nRules:")
    print(f"- precision: {report['rules']['precision']:.3f}")
    print(f"- recall   : {report['rules']['recall']:.3f}")
    print(f"- f1       : {report['rules']['f1']:.3f}")
    print("\nContexts:")
    print(f"- precision: {report['contexts']['precision']:.3f}")
    print(f"- recall   : {report['contexts']['recall']:.3f}")
    print(f"- f1       : {report['contexts']['f1']:.3f}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print("-", w)

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nWrote report to {args.report_out}")


if __name__ == "__main__":
    main()
