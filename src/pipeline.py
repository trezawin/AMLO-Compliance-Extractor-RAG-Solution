import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import numpy as np
from dotenv import load_dotenv
from rich.console import Console

# Silence HF tokenizer fork warning in CLI/batch runs.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from sentence_transformers import SentenceTransformer

from .config import paths, settings
from .prompts import build_prompt

load_dotenv()
console = Console()


@dataclass
class Retrieved:
    chunk: Dict
    score: float
    source: str


def load_embeddings(index_path: Path) -> Tuple[np.ndarray, List[Dict]]:
    embeddings = np.load(index_path)
    meta_path = index_path.with_suffix(".meta.jsonl")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata next to index: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]
    return embeddings, metadata


def load_bm25(bm25_path: Path):
    with bm25_path.open("rb") as f:
        payload = pickle.load(f)
    return payload["bm25"], payload["corpus"], payload["chunks"]


def normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo == 0:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class Retriever:
    def __init__(self, index_path: Path = paths.default_index, bm25_path: Path = paths.bm25_store):
        self.embeddings, self.meta = load_embeddings(index_path)
        self.bm25, self.corpus, self.bm25_chunks = load_bm25(bm25_path)
        self.embed_model = SentenceTransformer(settings.embedding_model, device=settings.device)

    def embed(self, text: str) -> np.ndarray:
        return self.embed_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)

    def search(self, query: str, k: int = settings.top_k) -> List[Retrieved]:
        # Embedding search (cosine via dot product; embeddings are normalized)
        query_vec = self.embed(query)  # shape (1, dim)
        sims = np.dot(self.embeddings, query_vec.T).flatten()  # (n,)
        top_idx = np.argpartition(-sims, min(k * 2, len(sims) - 1))[: k * 2]
        top_scores = sims[top_idx].tolist()
        embed_norm = normalize(top_scores)

        # BM25 search
        bm25_scores = self.bm25.get_scores(query.lower().split())
        top_bm25 = sorted(list(enumerate(bm25_scores)), key=lambda x: x[1], reverse=True)[: k * 2]
        bm25_norm = normalize([score for _, score in top_bm25])

        combined: Dict[int, float] = {}
        # merge embedding scores
        for pos, idx in enumerate(top_idx):
            combined[idx] = combined.get(idx, 0.0) + settings.embedding_weight * embed_norm[pos]

        # merge bm25
        for pos, (bm_idx, _) in enumerate(top_bm25):
            combined[bm_idx] = combined.get(bm_idx, 0.0) + settings.bm25_weight * bm25_norm[pos]

        # Rank
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        results: List[Retrieved] = []
        for idx, score in ranked:
            results.append(Retrieved(chunk=self.meta[idx], score=score, source="hybrid"))
        return results


def render_context(results: List[Retrieved]) -> str:
    parts = []
    for item in results:
        section = item.chunk.get("section")
        heading = item.chunk.get("heading")
        prefix = ""
        if section and heading:
            prefix = f"[{section}] {heading}"
        elif section:
            prefix = f"[{section}]"
        elif heading:
            prefix = heading
        parts.append(f"{prefix}\n{item.chunk['text']}".strip())
    return "\n\n---\n\n".join(parts)


def call_llm(prompt: str) -> Optional[str]:
    if settings.llm_provider.lower() == "none":
        console.print("[yellow]LLM provider set to 'none'; skipping call.")
        return None

    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[yellow]OPENAI_API_KEY not set; returning prompt only.")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        console.print("[red]openai package missing; install requirements or capture the prompt manually.")
        return None

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def parse_extraction_response(raw: Optional[str]) -> Dict[str, List[Dict]]:
    """
    Normalize the LLM response into a dict with "rules" and "contexts" lists.
    Accepts either:
      - the new object shape {"rules": [...], "contexts": [...]}, or
      - a legacy top-level list of rules (treated as {"rules": list, "contexts": []}).
    """
    empty = {"rules": [], "contexts": []}
    if not raw:
        return empty
    cleaned = raw.strip()
    if not cleaned:
        return empty
    # Strip simple fences if present.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        data = json.loads(cleaned)
    except Exception:
        return empty

    if isinstance(data, dict):
        rules = data.get("rules") or []
        contexts = data.get("contexts") or []
        if not isinstance(rules, list):
            rules = []
        if not isinstance(contexts, list):
            contexts = []
        return {"rules": [r for r in rules if isinstance(r, dict)], "contexts": [c for c in contexts if isinstance(c, dict)]}
    if isinstance(data, list):
        # Legacy: treat as rules-only list.
        return {"rules": [r for r in data if isinstance(r, dict)], "contexts": []}
    return empty


def run_extraction(query: str, top_k: int, dry_run: bool = False) -> Dict:
    retriever = Retriever()
    results = retriever.search(query, k=top_k)
    context = render_context(results)
    prompt = build_prompt(context=context, query=query)
    response = None if dry_run else call_llm(prompt)
    parsed = parse_extraction_response(response) if response else {"rules": [], "contexts": []}
    if parsed:
        # Normalize response to the canonical object shape.
        response = json.dumps(parsed, ensure_ascii=False, indent=2)
    return {
        "query": query,
        "top_k": top_k,
        "prompt": prompt,
        "response": response,
        "rules": parsed.get("rules", []),
        "contexts": parsed.get("contexts", []),
        "retrieved": [r.chunk for r in results],
    }


def main():
    parser = argparse.ArgumentParser(description="Run retrieval + prompt assembly (optional LLM call).")
    parser.add_argument("--query", required=True, help="Extraction topic or section focus")
    parser.add_argument("--top_k", type=int, default=settings.top_k, help="How many chunks to retrieve")
    parser.add_argument("--dry_run", action="store_true", help="Skip LLM call and only emit the prompt")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save JSON response")
    args = parser.parse_args()

    payload = run_extraction(query=args.query, top_k=args.top_k, dry_run=args.dry_run)
    if payload["response"]:
        console.print("[bold green]LLM response:")
        console.print(payload["response"])
    else:
        console.print("[cyan]Prompt (no LLM call performed):")
        console.print(payload["prompt"])

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        console.print(f"[bold green]Saved output to {args.save}")


if __name__ == "__main__":
    main()
