from typing import Any, Dict, List, Optional
import json

from fastapi import FastAPI
from pydantic import BaseModel

from .pipeline import Retriever, call_llm, render_context, parse_extraction_response
from .config import settings
from .prompts import build_prompt, ROLE_PROMPT, TASK_INSTRUCTION

app = FastAPI(title="AMLO RAG Extractor")
retriever = Retriever()


class ExtractRequest(BaseModel):
    query: str
    top_k: int = settings.top_k
    dry_run: bool = False


class RetrievedChunk(BaseModel):
    id: int
    text: str
    section: Optional[str] = None
    heading: Optional[str] = None
    source: Optional[str] = None
    tokens: Optional[int] = None


class ExtractResponse(BaseModel):
    query: str
    prompt: str
    response: Optional[str]
    retrieved: List[RetrievedChunk]
    rules: Optional[List[Dict[str, Any]]] = None
    contexts: Optional[List[Dict[str, Any]]] = None


@app.post("/extract", response_model=ExtractResponse)
def extract(body: ExtractRequest) -> ExtractResponse:
    results = retriever.search(body.query, k=body.top_k)
    context = render_context(results)
    prompt = build_prompt(context=context, query=body.query)
    response = None if body.dry_run else call_llm(prompt)
    parsed = {"rules": [], "contexts": []}
    if response:
        try:
            parsed = parse_extraction_response(response)
            response = json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return ExtractResponse(
        query=body.query,
        prompt=prompt,
        response=response,
        retrieved=[RetrievedChunk(**r.chunk) for r in results],
        rules=parsed.get("rules", []),
        contexts=parsed.get("contexts", []),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.embedding_model}
