import json
from collections import Counter
chunks = [json.loads(l) for l in open("data/processed/cap615.jsonl")]
payload = json.loads(open("data/processed/all_rules.json").read())
rules = payload.get("rules", []) if isinstance(payload, dict) else payload
section_chunks = Counter((c.get("section"), c.get("heading")) for c in chunks)
section_rules = Counter((r.get("section"), r.get("heading")) for r in rules)
print("Sections with chunks:", len(section_chunks))
print("Sections with rules:", len(section_rules))
print("\nMissing (no rules generated):")
for k in section_chunks:
    if k not in section_rules:
        print(" ", k)
