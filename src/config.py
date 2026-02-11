from pathlib import Path
from typing import Optional
from pydantic import BaseModel


class Paths(BaseModel):
    """Centralize project paths."""

    root: Path = Path(__file__).resolve().parent.parent
    raw_dir: Path = root / "data" / "raw"
    processed_dir: Path = root / "data" / "processed"
    default_chunks: Path = processed_dir / "cap615.jsonl"
    default_index: Path = processed_dir / "embeddings.npy"
    bm25_store: Path = processed_dir / "bm25.pkl"


class Settings(BaseModel):
    """Configurable knobs for ingestion and retrieval."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    max_chunk_tokens: int = 1200
    min_chunk_tokens: int = 200
    overlap_tokens: int = 80
    top_k: int = 8
    bm25_weight: float = 0.35
    embedding_weight: float = 0.65
    llm_provider: str = "openai"  # swap to "none" to skip calls
    llm_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None


paths = Paths()
settings = Settings()
