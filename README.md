# AMLO RAG Extractor

RAG pipeline scaffold to extract enforceable obligations from the HKMA Anti‑Money Laundering and Counter‑Financing of Terrorism Ordinance (Cap. 615).

## What’s here
- `src/ingest.py`: PDF/text ingestion and section‑aware chunking.
- `src/index.py`: Embedding + BM25 indexing (numpy embeddings + metadata).
- `src/pipeline.py`: Retrieval orchestration and prompt assembly.
- `src/prompts.py`: System/task prompts for the auditor role and JSON schema.
- `src/server.py`: FastAPI service for `/extract`.
- `src/config.py`: Paths and model configurables.
- `requirements.txt`: Python deps (install in a virtualenv).
- `data/raw`: Put the AMLO (Cap. 615) PDF or RTF cleaned text here.
- `data/processed`: Generated structured text, chunks, and vector index cache.

## Quick start
1) Create a virtualenv and install deps:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2) Add the AMLO source into `data/raw/`, e.g., `cap615.pdf`, `cap615.rtf`, or `cap615.txt`.  
   The ingestion code is configured to keep **only**:
   - Part 5B, sections 53ZRD – 53ZRL — Licensing and supervision of Virtual Asset Service Providers (VASPs)
   - Schedule 2 — Customer due diligence (CDD), record-keeping, and AML/CFT procedural requirements
   - Schedule 3B — Fit-and-proper criteria for VASP license applicants

3) Ingest + index:
```bash
python -m src.ingest --source data/raw/cap615.pdf --out data/processed/cap615.jsonl
python -m src.index --chunks data/processed/cap615.jsonl --index_out data/processed/embeddings.npy --bm25_out data/processed/bm25.pkl
```
If you’re on macOS with M1/M2 and hit GPU-related crashes, set CPU mode in `src/config.py` (`device="cpu"`).

4) Run an extraction (prints the prompt & optional LLM call):
```bash
python -m src.pipeline --query "customer due diligence obligations" --top_k 8
```
Provide `OPENAI_API_KEY` (or update `LLM_PROVIDER` in `src/pipeline.py`) to actually call a model. Without a key, the script will emit the assembled prompt for manual/other use.

5) Extract all sections to one JSON file:
```bash
python -m src.batch_extract --chunks data/processed/cap615.jsonl --out data/processed/all_rules.json
```
Use `--limit 5 --dry_run` to test quickly; set `OPENAI_API_KEY` before running to perform real calls.
Use `--offset`/`--count` to split runs into parts (e.g., run 60 contexts at a time).

Top 5 rules for a single query:
```bash
python scripts/extract_top5_rules.py --query "customer due diligence obligations" --top_k 8 --out data/processed/top5_rules.json
```

6) Summarize chunks (for inspection):
```bash
python scripts/summarize_chunks.py --chunks data/processed/cap615.jsonl --show 3
```

7) Provision coverage checks (target scope sanity):
```bash
python scripts/check_part5b_scope.py --chunks data/processed/cap615.jsonl
python scripts/check_schedule_scope.py --chunks data/processed/cap615.jsonl
```

8) Coverage sanity check (does chunking match the PDF):
```bash
python scripts/coverage_check.py --pdf data/raw/cap-615.pdf --chunks data/processed/cap615.jsonl --show_missing 10
```

5) Serve an API:
```bash
uvicorn src.server:app --reload --port 8000
```
Then POST to `/extract` with `{"query": "...", "top_k": 8}`.

## Common pitfalls
- If you hit `Client.__init__() got an unexpected keyword argument 'proxies'` from `openai`, upgrade the HTTP stack in your venv:  
  `pip install --upgrade "openai>=1.44.0" "httpx>=0.25,<0.28" "httpcore>=1.0.0"`
- Ensure `OPENAI_API_KEY` is set (shell export or `.env`) before running commands without `--dry_run`.
## Notes
- Chunker preserves section headings and citations; tune `--min_tokens/--max_tokens` as needed.
- Hybrid retrieval combines semantic + BM25; adjust weights in `src/pipeline.py`.
- Outputs enforce the JSON shape requested in the role instruction. Add schema validation in `pipeline.run_extraction` if desired.***
