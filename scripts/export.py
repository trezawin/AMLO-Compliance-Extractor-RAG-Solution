import json
import pandas as pd

# Path to your JSON file
input_path = "data/processed/all_rules.json"
output_path = "data/processed/all_rules.xlsx"

# Load JSON
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract rules and contexts
rules = data.get("rules", [])
contexts = data.get("contexts", [])

# Convert to DataFrames
df_rules = pd.DataFrame(rules)
df_contexts = pd.DataFrame(contexts)

# Export to Excel with separate sheets
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df_rules.to_excel(writer, sheet_name="Rules", index=False)
    df_contexts.to_excel(writer, sheet_name="Contexts", index=False)

print(f"Excel file successfully created at: {output_path}")