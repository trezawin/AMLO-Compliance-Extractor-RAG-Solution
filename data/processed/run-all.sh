cd /Users/treza/Documents/RAG-RuleAuthor
source .venv/bin/activate

# 1) Ingest scoped provisions from cap615.pdf
python -m src.ingest --source data/raw/cap615.pdf --out data/processed/cap615.jsonl

# 2) Validate scope coverage
python scripts/check_part5b_scope.py --chunks data/processed/cap615.jsonl
python scripts/check_schedule_scope.py --chunks data/processed/cap615.jsonl

# 3) Build retrieval index
python -m src.index \
  --chunks data/processed/cap615.jsonl \
  --index_out data/processed/embeddings.npy \
  --bm25_out data/processed/bm25.pkl

# 4) Validate index consistency
python scripts/check_index_consistency.py --strict

# 5) Run full batch extraction (rules + contexts)
python -m src.batch_extract \
  --chunks data/processed/cap615.jsonl \
  --out data/processed/all_rules.json

# 6) Run only top 5 rules for a single query
python scripts/extract_top5_rules.py \
  --query "customer due diligence obligations" \
  --top_k 8 \
  --out data/processed/top5_rules.json

# dry run
python -m src.batch_extract --chunks data/processed/cap615.jsonl --out data/processed/all_rules.json --limit 2 --dry_run
python scripts/extract_top5_rules.py --query "customer due diligence obligations" --top_k 8 --dry_run --out data/processed/top5_rules.json
